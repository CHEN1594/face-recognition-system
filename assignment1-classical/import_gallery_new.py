import re
from pathlib import Path

import cv2
import numpy as np

# =========================
# Tunable Parameters
# =========================
INPUT_DIR = Path("../gallery_new")
OUTPUT_DIR = Path("../gallery_preprocessed")
FACE_SIZE = (90, 90)
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_FACE_SIZE = 90

# Basic quality filtering (set to 0 to disable)
MIN_STD = 8.0
MIN_LAPLACIAN_VAR = 20.0

# Optional manual override for cascade XML
CASCADE_PATH_OVERRIDE = None
# Optional subset, e.g. ["Dominic", "SAI"]; set to [] to process all.
PROCESS_ONLY: list[str] = []

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_cascade_path() -> Path:
    if CASCADE_PATH_OVERRIDE is not None:
        p = Path(CASCADE_PATH_OVERRIDE)
        if p.exists():
            return p
        raise FileNotFoundError(f"CASCADE_PATH_OVERRIDE not found: {p}")

    local_xml = Path(__file__).resolve().parent / "haarcascade_frontalface_default.xml"
    if local_xml.exists():
        return local_xml

    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        p = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if p.exists():
            return p

    p = Path(cv2.__file__).resolve().parent / "data" / "haarcascade_frontalface_default.xml"
    if p.exists():
        return p

    raise FileNotFoundError("Cannot find haarcascade_frontalface_default.xml")


def preprocess_face(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = clahe.apply(gray)
    normalized = cv2.normalize(processed.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return (normalized * 255.0).astype(np.uint8)


def is_low_quality(image: np.ndarray) -> bool:
    std_val = float(np.std(image))
    lap_var = float(cv2.Laplacian(image, cv2.CV_64F).var())
    return std_val < MIN_STD or lap_var < MIN_LAPLACIAN_VAR


def extract_next_index(person_dir: Path, person_name: str) -> int:
    pattern = re.compile(rf"^{re.escape(person_name)}_(\d+)\.jpg$", re.IGNORECASE)
    max_idx = -1
    for p in person_dir.glob("*.jpg"):
        m = pattern.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1


def largest_face(faces: np.ndarray) -> tuple[int, int, int, int]:
    if len(faces) == 0:
        raise ValueError("No faces detected")
    x, y, w, h = max(faces, key=lambda b: int(b[2]) * int(b[3]))
    return int(x), int(y), int(w), int(h)


def sorted_images(person_input_dir: Path) -> list[Path]:
    files = [p for p in person_input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
    return sorted(files, key=lambda p: p.name.lower())


def imread_unicode(path: Path) -> np.ndarray | None:
    # cv2.imread can fail on non-ASCII Windows paths in some builds.
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cascade_path = resolve_cascade_path()
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")

    people = [p for p in sorted(INPUT_DIR.iterdir()) if p.is_dir()]
    if PROCESS_ONLY:
        allow = set(PROCESS_ONLY)
        people = [p for p in people if p.name in allow]
    if not people:
        print(f"No person folders found in {INPUT_DIR}")
        return

    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Cascade: {cascade_path}")
    print("-" * 80)

    total_in = 0
    total_saved = 0
    total_no_face = 0
    total_low_quality = 0
    total_read_error = 0

    for person_input_dir in people:
        person_name = person_input_dir.name
        person_output_dir = OUTPUT_DIR / person_name
        person_output_dir.mkdir(parents=True, exist_ok=True)
        next_index = extract_next_index(person_output_dir, person_name)

        person_in = 0
        person_saved = 0
        person_no_face = 0
        person_low_quality = 0
        person_read_error = 0

        for img_path in sorted_images(person_input_dir):
            total_in += 1
            person_in += 1

            img = imread_unicode(img_path)
            if img is None:
                total_read_error += 1
                person_read_error += 1
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=SCALE_FACTOR,
                minNeighbors=MIN_NEIGHBORS,
                minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
            )

            if len(faces) == 0:
                total_no_face += 1
                person_no_face += 1
                continue

            x, y, w, h = largest_face(faces)
            face_crop = gray[y : y + h, x : x + w]
            face_crop = cv2.resize(face_crop, FACE_SIZE, interpolation=cv2.INTER_AREA)
            out_img = preprocess_face(face_crop)

            if is_low_quality(out_img):
                total_low_quality += 1
                person_low_quality += 1
                continue

            out_name = f"{person_name}_{next_index}.jpg"
            out_path = person_output_dir / out_name
            cv2.imwrite(str(out_path), out_img)

            next_index += 1
            total_saved += 1
            person_saved += 1

        print(
            f"{person_name:<12} in={person_in:<3} saved={person_saved:<3} "
            f"no_face={person_no_face:<3} low_q={person_low_quality:<3} read_err={person_read_error:<3}"
        )

    print("-" * 80)
    print(
        f"Total in={total_in}, saved={total_saved}, no_face={total_no_face}, "
        f"low_q={total_low_quality}, read_err={total_read_error}"
    )
    print("Done.")


if __name__ == "__main__":
    main()
