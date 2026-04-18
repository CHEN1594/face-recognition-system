import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from facenet_pytorch import MTCNN, InceptionResnetV1


# =========================
# Tunable Parameters
# =========================
DATASET_DIR = Path("dataset/train")
OUTPUT_CKPT = Path("../models/facenet_partial_finetune.pth")
OUTPUT_META = Path("../models/facenet_partial_finetune_meta.json")

BATCH_SIZE = 16
EPOCHS = 8
LR = 1e-4
WEIGHT_DECAY = 1e-5
VAL_SPLIT = 0.2
SEED = 42


def load_rgb(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def iter_person_images(root: Path):
    persons = sorted([d for d in root.iterdir() if d.is_dir()])
    for label_idx, person_dir in enumerate(persons):
        files = sorted([p for p in person_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        yield label_idx, person_dir.name, files


def stratified_split(labels: np.ndarray, val_split: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for cls in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_split))
        if len(idx) >= 5:
            n_val = max(1, n_val)
        else:
            n_val = 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def build_backbone_state_dict(state_dict: dict):
    # Keep all shared backbone parameters; remove classifier-only head.
    return {k: v for k, v in state_dict.items() if not k.startswith("logits.")}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not DATASET_DIR.exists():
        print(f"Dataset folder not found: {DATASET_DIR}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        post_process=True,
        device=device,
    )

    class_names = []
    face_tensors = []
    label_list = []
    skipped = 0

    for label_idx, person_name, image_files in iter_person_images(DATASET_DIR):
        class_names.append(person_name)
        print(f"\nCollecting aligned faces: {person_name} ({len(image_files)} images)")
        for p in image_files:
            rgb = load_rgb(p)
            if rgb is None:
                skipped += 1
                continue
            face = mtcnn(rgb)
            if face is None:
                skipped += 1
                continue
            face_tensors.append(face.cpu())
            label_list.append(label_idx)

    if not face_tensors:
        print("No aligned faces found. Stop.")
        return

    x = torch.stack(face_tensors, dim=0)
    y = torch.tensor(label_list, dtype=torch.long)
    labels_np = y.numpy()
    num_classes = len(class_names)

    train_idx, val_idx = stratified_split(labels_np, VAL_SPLIT, SEED)
    if len(train_idx) == 0:
        print("Train split is empty. Stop.")
        return

    train_ds = TensorDataset(x[train_idx], y[train_idx])
    val_ds = TensorDataset(x[val_idx], y[val_idx]) if len(val_idx) > 0 else None

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) if val_ds else None

    model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=num_classes).to(device)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.last_linear.parameters():
        p.requires_grad = True
    for p in model.last_bn.parameters():
        p.requires_grad = True
    for p in model.logits.parameters():
        p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    print("\nTraining with partial unfreeze: last_linear + last_bn + logits")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            pred = torch.argmax(logits, dim=1)
            running_correct += int((pred == yb).sum().item())
            running_total += int(yb.size(0))

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)

        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    pred = torch.argmax(logits, dim=1)
                    val_correct += int((pred == yb).sum().item())
                    val_total += int(yb.size(0))
            val_acc = val_correct / max(1, val_total)
        else:
            val_acc = train_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

    if best_state is None:
        print("No best state captured. Stop.")
        return

    backbone_state = build_backbone_state_dict(best_state)
    OUTPUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone_state_dict": backbone_state,
            "class_names": class_names,
            "best_val_acc": best_val_acc,
            "trainable_layers": ["last_linear", "last_bn", "logits"],
            "epochs": EPOCHS,
        },
        str(OUTPUT_CKPT),
    )

    meta = {
        "num_classes": num_classes,
        "class_names": class_names,
        "samples_total": int(len(y)),
        "samples_train": int(len(train_idx)),
        "samples_val": int(len(val_idx)),
        "skipped_images": int(skipped),
        "best_val_acc": float(best_val_acc),
        "trainable_layers": ["last_linear", "last_bn", "logits"],
        "checkpoint": str(OUTPUT_CKPT),
    }
    with OUTPUT_META.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved:")
    print(f"  {OUTPUT_CKPT}")
    print(f"  {OUTPUT_META}")


if __name__ == "__main__":
    main()
