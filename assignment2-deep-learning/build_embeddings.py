import os
import cv2
import json
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

DATASET_DIR = "dataset/train"
OUTPUT_DIR = "embeddings"
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
LABELS_FILE = os.path.join(OUTPUT_DIR, "labels.npy")
CLASS_MAP_FILE = os.path.join(OUTPUT_DIR, "class_map.json")

def load_image_bgr(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        post_process=True,
        device=device
    )

    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    embeddings = []
    labels = []
    class_names = []

    persons = sorted([d for d in os.listdir(DATASET_DIR)
                      if os.path.isdir(os.path.join(DATASET_DIR, d))])

    if not persons:
        print("No person folders found in dataset/train.")
        return

    for label_idx, person in enumerate(persons):
        class_names.append(person)
        person_dir = os.path.join(DATASET_DIR, person)
        image_files = [f for f in os.listdir(person_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"\nProcessing {person} ({len(image_files)} images)")
        for img_name in image_files:
            img_path = os.path.join(person_dir, img_name)
            try:
                rgb = load_image_bgr(img_path)

                # Get aligned face tensor [3, 160, 160]
                face = mtcnn(rgb)
                if face is None:
                    print(f"  Skipped (no face): {img_name}")
                    continue

                face = face.unsqueeze(0).to(device)  # [1, 3, 160, 160]
                with torch.no_grad():
                    emb = resnet(face).cpu().numpy()[0]

                embeddings.append(emb)
                labels.append(label_idx)
                print(f"  OK: {img_name}")

            except Exception as e:
                print(f"  Error on {img_name}: {e}")

    if not embeddings:
        print("No embeddings created.")
        return

    embeddings = np.array(embeddings, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    np.save(EMBEDDINGS_FILE, embeddings)
    np.save(LABELS_FILE, labels)

    class_map = {str(i): name for i, name in enumerate(class_names)}
    with open(CLASS_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)

    print("\nSaved:")
    print(f"  {EMBEDDINGS_FILE}")
    print(f"  {LABELS_FILE}")
    print(f"  {CLASS_MAP_FILE}")
    print(f"Total embeddings: {len(embeddings)}")

if __name__ == "__main__":
    main()