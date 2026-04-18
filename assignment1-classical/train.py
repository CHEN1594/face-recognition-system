import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class FaceDataset:
    x: np.ndarray
    y: np.ndarray
    class_names: list[str]


def load_dataset(data_dir: Path) -> FaceDataset:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    class_dirs = [p for p in sorted(data_dir.iterdir()) if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found in {data_dir}")

    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    class_names = [p.name for p in class_dirs]

    for class_id, class_dir in enumerate(class_dirs):
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            if img.shape != (90, 90):
                img = cv2.resize(img, (90, 90), interpolation=cv2.INTER_AREA)

            face_vector = img.astype(np.float32).reshape(-1) / 255.0
            x_list.append(face_vector)
            y_list.append(class_id)

    if not x_list:
        raise ValueError(f"No valid images found in {data_dir}")

    return FaceDataset(
        x=np.vstack(x_list),
        y=np.array(y_list, dtype=np.int32),
        class_names=class_names,
    )


def build_model(n_classes: int, pca_components: float, n_neighbors: int) -> Pipeline:
    # LDA output dimension must be <= number of classes - 1
    lda_components = max(1, n_classes - 1)

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_components, svd_solver="full", whiten=True, random_state=42)),
            ("lda", LinearDiscriminantAnalysis(n_components=lda_components)),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean")),
        ]
    )


def print_dataset_summary(ds: FaceDataset) -> None:
    print("Dataset summary")
    print("-" * 40)
    print(f"Total images : {len(ds.y)}")
    print(f"Classes      : {len(ds.class_names)}")
    for class_id, class_name in enumerate(ds.class_names):
        count = int(np.sum(ds.y == class_id))
        print(f"{class_name:<15} {count}")
    print("-" * 40)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classical face recognizer (PCA + LDA + 1NN).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../gallery_preprocessed"),
        help="Preprocessed gallery directory (default: ../gallery_preprocessed)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("../models/pca_lda_knn.joblib"),
        help="Output model file path (default: ../models/pca_lda_knn.joblib)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test split ratio (default: 0.25)",
    )
    parser.add_argument(
        "--pca-components",
        type=float,
        default=0.98,
        help="PCA components; float keeps variance ratio (default: 0.98)",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=1,
        help="K for KNN classifier (default: 1)",
    )
    args = parser.parse_args()

    ds = load_dataset(args.data_dir)
    print_dataset_summary(ds)

    x_train, x_test, y_train, y_test = train_test_split(
        ds.x,
        ds.y,
        test_size=args.test_size,
        random_state=42,
        stratify=ds.y,
    )

    model = build_model(
        n_classes=len(ds.class_names),
        pca_components=args.pca_components,
        n_neighbors=args.neighbors,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print("Evaluation (hold-out test set)")
    print("-" * 40)
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=ds.class_names, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "class_names": ds.class_names,
        "image_size": (90, 90),
        "preprocess_hint": "Input faces should be grayscale, 90x90, CLAHE + minmax normalization compatible.",
    }
    joblib.dump(artifact, args.model_path)
    print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    main()
