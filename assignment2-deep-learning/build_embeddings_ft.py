import json
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


# =========================
# Tunable Parameters
# =========================
DATASET_DIR = Path("dataset/train")
OUTPUT_DIR = Path("embeddings_ft")
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npy"
LABELS_FILE = OUTPUT_DIR / "labels.npy"
CLASS_MAP_FILE = OUTPUT_DIR / "class_map.json"

FACENET_FT_CKPT = Path("../models/facenet_partial_finetune.pth")


def load_image_rgb(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    if not FACENET_FT_CKPT.exists():
        print(f"Fine-tuned checkpoint not found: {FACENET_FT_CKPT}")
        print("Run finetune_facenet.py first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        post_process=True,
        device=device,
    )

    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    ckpt = torch.load(str(FACENET_FT_CKPT), map_location=device)
    backbone_state = ckpt.get("backbone_state_dict", ckpt)
    resnet.load_state_dict(backbone_state, strict=False)

    embeddings = []
    labels = []
    class_names = []

    persons = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    if not persons:
        print("No person folders found in dataset/train.")
        return

    for label_idx, person_dir in enumerate(persons):
        class_names.append(person_dir.name)
        image_files = sorted([p for p in person_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        print(f"\nProcessing {person_dir.name} ({len(image_files)} images)")

        for img_path in image_files:
            try:
                rgb = load_image_rgb(img_path)
                face = mtcnn(rgb)
                if face is None:
                    print(f"  Skipped (no face): {img_path.name}")
                    continue

                face = face.unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = resnet(face).cpu().numpy()[0]

                embeddings.append(emb)
                labels.append(label_idx)
                print(f"  OK: {img_path.name}")
            except Exception as e:
                print(f"  Error on {img_path.name}: {e}")

    if not embeddings:
        print("No embeddings created.")
        return

    embeddings = np.array(embeddings, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    np.save(str(EMBEDDINGS_FILE), embeddings)
    np.save(str(LABELS_FILE), labels)

    class_map = {str(i): name for i, name in enumerate(class_names)}
    with CLASS_MAP_FILE.open("w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)

    print("\nSaved:")
    print(f"  {EMBEDDINGS_FILE}")
    print(f"  {LABELS_FILE}")
    print(f"  {CLASS_MAP_FILE}")
    print(f"Total embeddings: {len(embeddings)}")


if __name__ == "__main__":
    main()
