from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
MODEL_PATH = Path("../models/pca_lda_knn_v2.joblib")
FACE_DB_PATH = Path("../models/face_db_from_classical.joblib")

CAMERA_INDEX = 0
UNKNOWN_THRESHOLD = 60.0
FACE_DB_THRESHOLD = 0.55

MIN_FACE_SIZE = 90
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5

SMOOTH_WINDOW = 12
SMOOTH_MIN_COUNT = 7
TRACK_MATCH_MAX_DIST = 120
TRACK_TTL_FRAMES = 20

# UI
APP_TITLE = "Face Recognition Studio"
APP_SUBTITLE = "Real-time identity recognition"
WINDOW_SIZE = "1320x860"
PRIMARY_ACCENT = "#4F8CFF"

MODEL_OPTIONS = ["knn", "embed"]
MODEL_LABEL_TO_BACKEND = {
    "knn": "knn",
    "embed": "face_db",
}

FACE_MODE_OPTIONS = ["Multi", "Single"]
SMOOTHING_OPTIONS = ["On", "Off"]


def preprocess_face(gray_face: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = clahe.apply(gray_face)
    normalized = cv2.normalize(processed.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return (normalized * 255.0).astype(np.uint8)


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)


def resolve_haar_cascade_path() -> str:
    filename = "haarcascade_frontalface_default.xml"
    candidates = []
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(str(Path(cv2.data.haarcascades) / filename))
    candidates.append(str(Path(__file__).resolve().parent / filename))
    candidates.append(filename)
    for p in candidates:
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"Cannot find '{filename}'.")


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
        if backend not in {"knn", "face_db"}:
            raise ValueError("backend must be 'knn' or 'face_db'")
        self.backend = backend
        self.model = None
        self.class_names: list[str] = []
        self.image_size = (90, 90)
        self.face_db = None
        self.face_cascade = None
        self._load()

    def _load(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        artifact = joblib.load(MODEL_PATH)
        self.model = artifact["model"]
        self.class_names = artifact["class_names"]
        self.image_size = tuple(artifact.get("image_size", (90, 90)))

        cascade_path = resolve_haar_cascade_path()
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade.")

        if self.backend == "face_db":
            if not FACE_DB_PATH.exists():
                raise FileNotFoundError(f"Face DB not found: {FACE_DB_PATH}")
            self.face_db = joblib.load(FACE_DB_PATH)

    def _predict_knn(self, face_vec: np.ndarray) -> tuple[str, float]:
        pred_id = int(self.model.predict(face_vec)[0])
        knn = self.model.named_steps["knn"]
        features = self.model[:-1].transform(face_vec)
        dist, _ = knn.kneighbors(features, n_neighbors=1)
        distance = float(dist[0][0])
        if distance > UNKNOWN_THRESHOLD:
            return "Unknown", distance
        return self.class_names[pred_id], distance

    def _predict_face_db(self, face_vec: np.ndarray) -> tuple[str, float]:
        transform_model = self.model[:-1]
        feat = transform_model.transform(face_vec).astype(np.float32)
        feat = _l2_normalize(feat)
        prototypes = self.face_db["prototypes"].astype(np.float32)
        names = self.face_db["names"]
        sims = np.dot(prototypes, feat[0])
        idx = int(np.argmax(sims))
        sim = float(sims[idx])
        if sim < FACE_DB_THRESHOLD:
            return "Unknown", sim
        return names[idx], sim

    def detect_and_recognize(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int, str, float]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
        )

        out = []
        for (x, y, w, h) in faces:
            crop = gray[y : y + h, x : x + w]
            resized = cv2.resize(crop, self.image_size, interpolation=cv2.INTER_AREA)
            preprocessed = preprocess_face(resized)
            vec = preprocessed.astype(np.float32).reshape(1, -1) / 255.0
            if self.backend == "face_db":
                name, score = self._predict_face_db(vec)
            else:
                name, score = self._predict_knn(vec)
            out.append((x, y, w, h, name, score))
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

        self.model_var = tk.StringVar(value="knn")
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
            if backend == "face_db":
                print(f"[Startup] Threshold: {FACE_DB_THRESHOLD:.2f} (embed)")
            else:
                print(f"[Startup] Threshold: {UNKNOWN_THRESHOLD:.2f} (knn distance)")
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
        self.track = TrackState(positions={}, histories={}, last_seen={}, next_id=0, frame_idx=0)
        self._init_engine()
        self.mode_chip_var.set(self.face_mode_var.get())
        self.smoothing_chip_var.set(self.smoothing_var.get())
        if was_running and self.engine is not None:
            self.start_camera()

    def start_camera(self) -> None:
        if self.running:
            return
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Cannot open camera index {CAMERA_INDEX}")
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

        stale_ids = [
            tid for tid, last in self.track.last_seen.items()
            if self.track.frame_idx - last > TRACK_TTL_FRAMES
        ]
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
