from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import gc
import importlib.util
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import joblib
import numpy as np

try:
    from PIL import Image, ImageTk
except Exception as e:
    raise ImportError("Pillow is required. Install with: pip install pillow") from e

# =========================
# Tunable Parameters
# =========================
EMBEDDINGS_FILE = Path("embeddings/embeddings.npy")
LABELS_FILE = Path("embeddings/labels.npy")
CLASS_MAP_FILE = Path("embeddings/class_map.json")
FT_EMBEDDINGS_FILE = Path("embeddings_ft/embeddings.npy")
FT_LABELS_FILE = Path("embeddings_ft/labels.npy")
FT_CLASS_MAP_FILE = Path("embeddings_ft/class_map.json")
EMBED_FACE_DB_PATH = Path("../models/face_db_embed.joblib")
FACENET_FT_CKPT = Path("../models/facenet_partial_finetune.pth")

CAMERA_INDEX = 0
MAX_CAMERA_INDEX = 5
THRESHOLD = 0.90  # lower = stricter, tune between about 0.8 and 1.1
FACENET_FT_THRESHOLD = 1.15  # fine-tuned model often needs a looser threshold
EMBED_FACE_DB_THRESHOLD = 0.35

EMBED_MODEL_NAME = "buffalo_l"
EMBED_DET_SIZE = (640, 640)
EMBED_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

MODEL_OPTIONS = ["FacenetDB", "FacenetFT", "InsightFace"]
MODEL_LABEL_TO_BACKEND = {
    "FacenetDB": "facenet_db",
    "FacenetFT": "facenet_ft_db",
    "InsightFace": "embed_face_db",
}
DEFAULT_MODEL_LABEL = os.environ.get("FRS_MODEL", "FacenetDB")
if DEFAULT_MODEL_LABEL == "EmbedDB":
    DEFAULT_MODEL_LABEL = "InsightFace"
if DEFAULT_MODEL_LABEL not in MODEL_LABEL_TO_BACKEND:
    DEFAULT_MODEL_LABEL = "FacenetDB"
# Prevent immediate abort on duplicated OpenMP runtime in some Windows envs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SMOOTH_WINDOW = 12
SMOOTH_MIN_COUNT = 7
TRACK_MATCH_MAX_DIST = 120
TRACK_TTL_FRAMES = 20

# UI
APP_TITLE = "Face Recognition Studio"
APP_SUBTITLE = "Real-time identity recognition"
WINDOW_SIZE = "1320x860"
PRIMARY_ACCENT = "#4F8CFF"
FACE_MODE_OPTIONS = ["Multi", "Single"]
SMOOTHING_OPTIONS = ["On", "Off"]


def open_camera_with_fallback(preferred_index: int = 0, max_index: int = 5):
    candidate_indices = [preferred_index] + [i for i in range(max_index + 1) if i != preferred_index]
    backends = [
        ("MSMF", cv2.CAP_MSMF),
        ("DSHOW", cv2.CAP_DSHOW),
        ("ANY", cv2.CAP_ANY),
    ]

    for backend_name, backend in backends:
        for idx in candidate_indices:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                print(f"Using camera index: {idx} (backend={backend_name})")
                return cap
            cap.release()
    return None


def load_facenet_stack():
    import torch  # local import to avoid mixed runtime load when not needed
    from facenet_pytorch import MTCNN, InceptionResnetV1

    return torch, MTCNN, InceptionResnetV1


def load_embed_backend():
    backend_path = Path(__file__).resolve().parent / "embedding_backend.py"
    print(f"[Startup] InsightFace backend file: {backend_path}")
    if not backend_path.exists():
        raise FileNotFoundError(f"embedding_backend.py not found: {backend_path}")

    spec = importlib.util.spec_from_file_location("a1_embedding_backend", str(backend_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from: {backend_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "FaceEmbedder"):
        raise ImportError(f"FaceEmbedder class not found in: {backend_path}")
    return module.FaceEmbedder


def draw_fancy_box(img: np.ndarray, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
    x2, y2 = x + w, y + h
    corner = max(12, int(min(w, h) * 0.12))
    th = 2
    cv2.rectangle(img, (x, y), (x2, y2), color, 1)
    cv2.line(img, (x, y), (x + corner, y), color, th)
    cv2.line(img, (x, y), (x, y + corner), color, th)
    cv2.line(img, (x2, y), (x2 - corner, y), color, th)
    cv2.line(img, (x2, y), (x2, y + corner), color, th)
    cv2.line(img, (x, y2), (x + corner, y2), color, th)
    cv2.line(img, (x, y2), (x, y2 - corner), color, th)
    cv2.line(img, (x2, y2), (x2 - corner, y2), color, th)
    cv2.line(img, (x2, y2), (x2, y2 - corner), color, th)


def draw_label_card(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.55
    thickness = 1
    (tw, th_text), _ = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 10, 8
    card_w = tw + pad_x * 2
    card_h = th_text + pad_y * 2
    card_x = max(8, x)
    card_y = max(8, y - card_h - 10)
    if card_x + card_w > img.shape[1] - 8:
        card_x = max(8, img.shape[1] - card_w - 8)

    overlay = img.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    cv2.rectangle(img, (card_x, card_y), (card_x + card_w, card_y + card_h), color, 1)
    cv2.putText(
        img,
        text,
        (card_x + pad_x, card_y + card_h - pad_y),
        font,
        scale,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA,
    )


@dataclass
class TrackState:
    positions: dict[int, tuple[float, float]]
    histories: dict[int, deque[str]]
    last_seen: dict[int, int]
    next_id: int
    frame_idx: int


class RecognitionEngine:
    def __init__(self, backend: str) -> None:
        if backend not in {"facenet_db", "facenet_ft_db", "embed_face_db"}:
            raise ValueError("backend must be 'facenet_db', 'facenet_ft_db', or 'embed_face_db'")
        self.backend = backend

        self.known_embeddings = None
        self.known_labels = None
        self.class_map = None
        self.device = "cpu"
        self.torch = None
        self.mtcnn = None
        self.resnet = None
        self.embed_db = None
        self.embedder = None
        self._load()

    def _load(self) -> None:
        if self.backend in {"facenet_db", "facenet_ft_db"}:
            torch, MTCNN, InceptionResnetV1 = load_facenet_stack()
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.backend == "facenet_db":
                emb_file, labels_file, class_map_file = EMBEDDINGS_FILE, LABELS_FILE, CLASS_MAP_FILE
            else:
                emb_file, labels_file, class_map_file = FT_EMBEDDINGS_FILE, FT_LABELS_FILE, FT_CLASS_MAP_FILE

            if not (emb_file.exists() and labels_file.exists() and class_map_file.exists()):
                if self.backend == "facenet_ft_db":
                    raise FileNotFoundError(
                        "Missing fine-tuned embedding files. "
                        "Run finetune_facenet.py then build_embeddings_ft.py first."
                    )
                raise FileNotFoundError("Missing embedding files. Run build_embeddings.py first.")

            self.known_embeddings = np.load(str(emb_file))
            self.known_labels = np.load(str(labels_file))
            with class_map_file.open("r", encoding="utf-8") as f:
                self.class_map = json.load(f)

            self.mtcnn = MTCNN(
                image_size=160,
                margin=20,
                keep_all=True,
                post_process=True,
                device=self.device,
            )
            self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            if self.backend == "facenet_ft_db":
                if not FACENET_FT_CKPT.exists():
                    raise FileNotFoundError(
                        f"Fine-tuned checkpoint not found: {FACENET_FT_CKPT}. "
                        "Run finetune_facenet.py first."
                    )
                ckpt = torch.load(str(FACENET_FT_CKPT), map_location=self.device)
                backbone_state = ckpt.get("backbone_state_dict", ckpt)
                self.resnet.load_state_dict(backbone_state, strict=False)
            return

        FaceEmbedder = load_embed_backend()
        self.device = "onnx"
        if not EMBED_FACE_DB_PATH.exists():
            raise FileNotFoundError(f"Embedding DB not found: {EMBED_FACE_DB_PATH}")

        self.embed_db = joblib.load(EMBED_FACE_DB_PATH)
        self.embedder = FaceEmbedder(
            model_name=EMBED_MODEL_NAME,
            det_size=EMBED_DET_SIZE,
            providers=EMBED_PROVIDERS,
            quiet=True,
        )

    def detect_and_recognize(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int, str, float]]:
        if self.backend == "embed_face_db":
            faces = self.embedder.get_faces(frame_bgr)
            prototypes = self.embed_db["prototypes"].astype(np.float32)
            names = self.embed_db["names"]
            out = []
            for f in faces:
                x1, y1, x2, y2 = f.bbox.astype(int).tolist()
                emb = f.embedding.astype(np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                sims = np.dot(prototypes, emb)
                idx = int(np.argmax(sims))
                sim = float(sims[idx])
                name = names[idx] if sim >= EMBED_FACE_DB_THRESHOLD else "Unknown"
                out.append((x1, y1, max(1, x2 - x1), max(1, y2 - y1), name, sim))
            return out

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn.detect(rgb)
        faces = self.mtcnn(rgb)
        if boxes is None or faces is None:
            return []

        if faces.ndim == 3:
            faces = faces.unsqueeze(0)
        faces = faces.to(self.device)

        with self.torch.no_grad():
            embeddings = self.resnet(faces).cpu().numpy()

        out = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            face_emb = embeddings[i]
            distances = np.linalg.norm(self.known_embeddings - face_emb, axis=1)
            min_idx = int(np.argmin(distances))
            min_dist = float(distances[min_idx])
            facenet_threshold = FACENET_FT_THRESHOLD if self.backend == "facenet_ft_db" else THRESHOLD
            if min_dist < facenet_threshold:
                label_id = int(self.known_labels[min_idx])
                name = self.class_map.get(str(label_id), "Unknown")
            else:
                name = "Unknown"
            out.append((x1, y1, max(1, x2 - x1), max(1, y2 - y1), name, min_dist))
        return out


class LiveFaceApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(WINDOW_SIZE)
        self.minsize(1100, 740)
        self.configure(bg="#0F1424")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = None
        self.running = False
        self.photo_ref = None
        self.engine = None
        self.fps = 0.0
        self.prev_tick = cv2.getTickCount()
        self.tick_freq = cv2.getTickFrequency()

        self.track = TrackState(positions={}, histories={}, last_seen={}, next_id=0, frame_idx=0)

        self.model_var = tk.StringVar(value=DEFAULT_MODEL_LABEL)
        self.current_model_label = self.model_var.get()
        self.face_mode_var = tk.StringVar(value="Multi")
        self.smoothing_var = tk.StringVar(value="On")
        self.status_var = tk.StringVar(value="Idle")
        self.faces_var = tk.StringVar(value="0")
        self.fps_var = tk.StringVar(value="0.0")
        self.method_chip_var = tk.StringVar(value=self.model_var.get())
        self.mode_chip_var = tk.StringVar(value=self.face_mode_var.get())
        self.smoothing_chip_var = tk.StringVar(value=self.smoothing_var.get())
        self.time_var = tk.StringVar(value="")

        self._build_styles()
        self._build_layout()
        self._tick_clock()
        self._init_engine()
        self.start_camera()

    def _build_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Root.TFrame", background="#0F1424")
        style.configure("Surface.TFrame", background="#141B2E")
        style.configure("Card.TFrame", background="#1A223A")
        style.configure("Sidebar.TFrame", background="#121A2D")
        style.configure("Title.TLabel", background="#0F1424", foreground="#F5F7FF", font=("Segoe UI Semibold", 19))
        style.configure("Subtitle.TLabel", background="#0F1424", foreground="#A8B3CF", font=("Segoe UI", 10))
        style.configure("HeaderTime.TLabel", background="#0F1424", foreground="#CBD5EE", font=("Segoe UI", 10))
        style.configure("ChipKey.TLabel", background="#1A223A", foreground="#9FB0D8", font=("Segoe UI", 9))
        style.configure("ChipVal.TLabel", background="#1A223A", foreground="#F2F6FF", font=("Segoe UI Semibold", 12))
        style.configure("Control.TButton", background=PRIMARY_ACCENT, foreground="white", padding=(14, 9))
        style.map("Control.TButton", background=[("active", "#6C9EFF"), ("pressed", "#3D79F4")])
        style.configure("Ghost.TButton", background="#212B47", foreground="#EAF0FF", padding=(14, 9))
        style.map("Ghost.TButton", background=[("active", "#2A3658"), ("pressed", "#1B243C")])
        style.configure(
            "Modern.TCombobox",
            fieldbackground="#1E2945",
            background="#1E2945",
            foreground="#F8FBFF",
            arrowcolor="#F8FBFF",
            selectbackground="#2F4D8A",
            selectforeground="#FFFFFF",
            bordercolor="#2B3A61",
            lightcolor="#2B3A61",
            darkcolor="#2B3A61",
        )
        style.map(
            "Modern.TCombobox",
            fieldbackground=[("readonly", "#1E2945")],
            foreground=[("readonly", "#F8FBFF")],
            selectbackground=[("readonly", "#2F4D8A")],
            selectforeground=[("readonly", "#FFFFFF")],
        )
        self.option_add("*TCombobox*Listbox.background", "#1E2945")
        self.option_add("*TCombobox*Listbox.foreground", "#F8FBFF")
        self.option_add("*TCombobox*Listbox.selectBackground", "#2F4D8A")
        self.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")

    def _build_layout(self) -> None:
        root = ttk.Frame(self, style="Root.TFrame", padding=18)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root, style="Root.TFrame")
        header.pack(fill="x")
        left = ttk.Frame(header, style="Root.TFrame")
        left.pack(side="left", fill="x", expand=True)
        ttk.Label(left, text=APP_TITLE, style="Title.TLabel").pack(anchor="w")
        ttk.Label(left, text=APP_SUBTITLE, style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))
        ttk.Label(header, textvariable=self.time_var, style="HeaderTime.TLabel").pack(side="right", anchor="ne", pady=4)

        chips_row = ttk.Frame(root, style="Root.TFrame", padding=(0, 14, 0, 12))
        chips_row.pack(fill="x")
        self._chip(chips_row, "METHOD", self.method_chip_var).pack(side="left", padx=(0, 12))
        self._chip(chips_row, "MODE", self.mode_chip_var).pack(side="left", padx=(0, 12))
        self._chip(chips_row, "SMOOTH", self.smoothing_chip_var).pack(side="left", padx=(0, 12))
        self._chip(chips_row, "FACES", self.faces_var).pack(side="left", padx=(0, 12))
        self._chip(chips_row, "FPS", self.fps_var).pack(side="left", padx=(0, 12))
        self._chip(chips_row, "STATUS", self.status_var).pack(side="left")

        body = ttk.Frame(root, style="Root.TFrame", padding=(0, 10, 0, 0))
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(body, style="Sidebar.TFrame", padding=14, width=280)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        camera_shell = ttk.Frame(body, style="Surface.TFrame", padding=12)
        camera_shell.pack(side="left", fill="both", expand=True, padx=(14, 0))
        camera_shell.pack_propagate(False)
        self.camera_frame = tk.Label(
            camera_shell,
            bg="#0D1220",
            bd=0,
            highlightthickness=1,
            highlightbackground="#26304B",
        )
        self.camera_frame.pack(fill="both", expand=True)

        ttk.Label(sidebar, text="Controls", style="Subtitle.TLabel").pack(anchor="w")
        ttk.Frame(sidebar, style="Sidebar.TFrame", height=8).pack(fill="x")

        start_stop_row = ttk.Frame(sidebar, style="Sidebar.TFrame")
        start_stop_row.pack(fill="x")
        ttk.Button(start_stop_row, text="Start", style="Control.TButton", command=self.start_camera).pack(side="left", fill="x", expand=True)
        ttk.Button(start_stop_row, text="Stop", style="Ghost.TButton", command=self.stop_camera).pack(side="left", fill="x", expand=True, padx=(10, 0))

        model_box = ttk.Frame(sidebar, style="Sidebar.TFrame")
        model_box.pack(fill="x", pady=(12, 0))
        ttk.Label(model_box, text="Model", style="Subtitle.TLabel").pack(anchor="w")
        self.model_combo = ttk.Combobox(
            model_box,
            state="readonly",
            textvariable=self.model_var,
            values=MODEL_OPTIONS,
            style="Modern.TCombobox",
            width=14,
        )
        self.model_combo.pack(fill="x", pady=(3, 0))

        mode_box = ttk.Frame(sidebar, style="Sidebar.TFrame")
        mode_box.pack(fill="x", pady=(12, 0))
        ttk.Label(mode_box, text="Face Mode", style="Subtitle.TLabel").pack(anchor="w")
        self.mode_combo = ttk.Combobox(
            mode_box,
            state="readonly",
            textvariable=self.face_mode_var,
            values=FACE_MODE_OPTIONS,
            style="Modern.TCombobox",
            width=14,
        )
        self.mode_combo.pack(fill="x", pady=(3, 0))

        smooth_box = ttk.Frame(sidebar, style="Sidebar.TFrame")
        smooth_box.pack(fill="x", pady=(12, 0))
        ttk.Label(smooth_box, text="Smoothing", style="Subtitle.TLabel").pack(anchor="w")
        self.smoothing_combo = ttk.Combobox(
            smooth_box,
            state="readonly",
            textvariable=self.smoothing_var,
            values=SMOOTHING_OPTIONS,
            style="Modern.TCombobox",
            width=14,
        )
        self.smoothing_combo.pack(fill="x", pady=(3, 0))

        ttk.Button(sidebar, text="Apply", style="Ghost.TButton", command=self.apply_settings).pack(fill="x", pady=(14, 0))
        ttk.Button(sidebar, text="Exit", style="Ghost.TButton", command=self.on_close).pack(fill="x", pady=(8, 0))

    def _chip(self, parent: ttk.Frame, key: str, value_var: tk.StringVar) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(14, 9))
        ttk.Label(card, text=key, style="ChipKey.TLabel").pack(anchor="w")
        ttk.Label(card, textvariable=value_var, style="ChipVal.TLabel").pack(anchor="w", pady=(1, 0))
        return card

    def _tick_clock(self) -> None:
        self.time_var.set(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.after(500, self._tick_clock)

    def _init_engine(self) -> None:
        try:
            self.status_var.set("Loading Model...")
            self.update_idletasks()
            backend = MODEL_LABEL_TO_BACKEND[self.model_var.get()]
            self.engine = RecognitionEngine(backend)
            self.current_model_label = self.model_var.get()
            self.status_var.set("Ready")
            self.method_chip_var.set(self.model_var.get())
            self.mode_chip_var.set(self.face_mode_var.get())
            self.smoothing_chip_var.set(self.smoothing_var.get())
            print(f"[Startup] Backend: {backend}")
            print(f"[Startup] Device: {self.engine.device}")
            if backend == "facenet_ft_db":
                print(f"[Startup] Threshold: {FACENET_FT_THRESHOLD:.2f} (FacenetFT)")
            elif backend == "facenet_db":
                print(f"[Startup] Threshold: {THRESHOLD:.2f} (FacenetDB)")
            print("[Startup] Recognition backend ready.")
        except Exception as e:
            self.status_var.set("Error")
            messagebox.showerror("Initialization Error", str(e))

    def apply_settings(self) -> None:
        was_running = self.running
        if was_running:
            self.stop_camera()
        if self.model_var.get() != self.current_model_label:
            self.engine = None
            gc.collect()
        self.track = TrackState(positions={}, histories={}, last_seen={}, next_id=0, frame_idx=0)
        self._init_engine()
        self.mode_chip_var.set(self.face_mode_var.get())
        self.smoothing_chip_var.set(self.smoothing_var.get())
        if was_running and self.engine is not None:
            self.start_camera()

    def start_camera(self) -> None:
        if self.running:
            return
        self.cap = open_camera_with_fallback(CAMERA_INDEX, max_index=MAX_CAMERA_INDEX)
        if self.cap is None:
            messagebox.showerror("Camera Error", "Could not open webcam. Please close other camera apps and retry.")
            return
        self.running = True
        self.status_var.set("Running")
        self.prev_tick = cv2.getTickCount()
        self._update_loop()

    def stop_camera(self) -> None:
        self.running = False
        self.status_var.set("Stopped")
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _assign_track_ids(self, detections: list[tuple[int, int, int, int, str, float]]) -> list[int]:
        remaining_tracks = set(self.track.positions.keys())
        assigned_ids: list[int] = []

        for (x, y, w, h, _, _) in detections:
            cx, cy = x + w * 0.5, y + h * 0.5
            best_tid = None
            best_d2 = None
            for tid in remaining_tracks:
                tx, ty = self.track.positions[tid]
                d2 = (cx - tx) ** 2 + (cy - ty) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_tid = tid

            if best_tid is not None and best_d2 is not None and best_d2 <= TRACK_MATCH_MAX_DIST * TRACK_MATCH_MAX_DIST:
                assigned_ids.append(best_tid)
                remaining_tracks.remove(best_tid)
            else:
                assigned_ids.append(self.track.next_id)
                self.track.next_id += 1

        return assigned_ids

    def _smooth_label(self, track_id: int, raw_name: str) -> str:
        if self.smoothing_var.get() == "Off":
            return raw_name
        hist = self.track.histories.setdefault(track_id, deque(maxlen=SMOOTH_WINDOW))
        hist.append(raw_name)
        counts = Counter(hist)
        label, count = counts.most_common(1)[0]
        if count >= SMOOTH_MIN_COUNT:
            return label
        return "Unknown"

    def _update_loop(self) -> None:
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set("Camera Read Failed")
            self.after(60, self._update_loop)
            return

        frame = cv2.flip(frame, 1)
        now_tick = cv2.getTickCount()
        dt = (now_tick - self.prev_tick) / self.tick_freq
        self.prev_tick = now_tick
        if dt > 0:
            current_fps = 1.0 / dt
            self.fps = current_fps if self.fps == 0.0 else (0.90 * self.fps + 0.10 * current_fps)

        detections = self.engine.detect_and_recognize(frame) if self.engine is not None else []
        if self.face_mode_var.get() == "Single" and detections:
            detections = [max(detections, key=lambda d: int(d[2]) * int(d[3]))]

        self.track.frame_idx += 1
        track_ids = self._assign_track_ids(detections)
        self.faces_var.set(str(len(detections)))
        self.fps_var.set(f"{self.fps:.1f}")

        for det, tid in zip(detections, track_ids):
            x, y, w, h, raw_name, _ = det
            self.track.positions[tid] = (x + w * 0.5, y + h * 0.5)
            self.track.last_seen[tid] = self.track.frame_idx
            name = self._smooth_label(tid, raw_name)

            color = (88, 222, 128) if name != "Unknown" else (93, 126, 255)
            draw_fancy_box(frame, x, y, w, h, color)
            draw_label_card(frame, name, x, y, color)

        stale_ids = [tid for tid, last in self.track.last_seen.items() if self.track.frame_idx - last > TRACK_TTL_FRAMES]
        for tid in stale_ids:
            self.track.last_seen.pop(tid, None)
            self.track.positions.pop(tid, None)
            self.track.histories.pop(tid, None)

        self._render_frame(frame)
        self.after(15, self._update_loop)

    def _render_frame(self, frame_bgr: np.ndarray) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        target_w = self.camera_frame.winfo_width()
        target_h = self.camera_frame.winfo_height()
        if target_w < 2 or target_h < 2:
            target_w, target_h = 960, 540

        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:, :, :] = (11, 16, 28)
        ox = (target_w - nw) // 2
        oy = (target_h - nh) // 2
        canvas[oy : oy + nh, ox : ox + nw] = resized

        image = Image.fromarray(canvas)
        self.photo_ref = ImageTk.PhotoImage(image=image)
        self.camera_frame.configure(image=self.photo_ref)

    def on_close(self) -> None:
        self.stop_camera()
        self.destroy()


def main() -> None:
    app = LiveFaceApp()
    app.mainloop()


if __name__ == "__main__":
    main()
