from pathlib import Path

import cv2
import joblib
import numpy as np

# =========================
# Tunable Parameters
# =========================
MODEL_PATH = Path("../models/pca_lda_knn_v2.joblib")
DATA_DIR = Path("../gallery_preprocessed")
OUTPUT_DB_PATH = Path("../models/face_db_from_classical.joblib")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    class_names = artifact["class_names"]
    image_size = tuple(artifact.get("image_size", (90, 90)))

    transform_model = model[:-1]  # scaler + pca + lda

    prototypes = []
    sample_counts = []
    valid_names = []

    print(f"Model : {MODEL_PATH}")
    print(f"Data  : {DATA_DIR}")
    print(f"Output: {OUTPUT_DB_PATH}")
    print("-" * 70)

    for person_name in class_names:
        person_dir = DATA_DIR / person_name
        if not person_dir.exists():
            print(f"{person_name:<12} missing folder, skipped")
            continue

        person_features = []
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != image_size:
                img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)

            vec = img.astype(np.float32).reshape(1, -1) / 255.0
            feat = transform_model.transform(vec)  # (1, d_lda)
            person_features.append(feat[0])

        if not person_features:
            print(f"{person_name:<12} no valid images, skipped")
            continue

        feats = np.asarray(person_features, dtype=np.float32)
        feats = l2_normalize(feats)
        proto = np.mean(feats, axis=0, keepdims=True)
        proto = l2_normalize(proto)[0]

        prototypes.append(proto)
        sample_counts.append(len(person_features))
        valid_names.append(person_name)
        print(f"{person_name:<12} samples={len(person_features)}")

    if not prototypes:
        raise RuntimeError("No prototypes were built. Please check your data.")

    db_artifact = {
        "names": valid_names,
        "prototypes": np.asarray(prototypes, dtype=np.float32),
        "sample_counts": sample_counts,
        "model_path": str(MODEL_PATH),
        "space": "classical_lda",
    }

    OUTPUT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(db_artifact, OUTPUT_DB_PATH)
    print("-" * 70)
    print(f"Face DB saved: {OUTPUT_DB_PATH}")


if __name__ == "__main__":
    main()
