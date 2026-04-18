from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io

import cv2
import numpy as np


class FaceEmbedder:
    """
    Thin wrapper around InsightFace FaceAnalysis.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        ctx_id: int = 0,
        providers: list[str] | None = None,
        quiet: bool = True,
    ) -> None:
        try:
            from insightface.app import FaceAnalysis
        except Exception as e:
            raise ImportError(
                "Failed to import insightface. Install with:\n"
                "  pip install insightface onnxruntime\n"
                "or conda/pip equivalent in your frs environment."
            ) from e

        if providers is None:
            providers = self._resolve_providers()
            ctx_id = 0 if "CUDAExecutionProvider" in providers else -1

        with self._suppress_library_output(enabled=quiet):
            self.app = FaceAnalysis(name=model_name, providers=providers)
            self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    @staticmethod
    @contextmanager
    def _suppress_library_output(enabled: bool):
        if not enabled:
            yield
            return
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            yield

    @staticmethod
    def _resolve_providers() -> list[str]:
        try:
            import onnxruntime as ort

            ort.set_default_logger_severity(3)
            available = set(ort.get_available_providers())
        except Exception:
            available = set()

        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        chosen = [p for p in preferred if p in available]
        if chosen:
            return chosen
        return ["CPUExecutionProvider"]

    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def get_faces(self, image: np.ndarray):
        bgr = self._to_bgr(image)
        return self.app.get(bgr)
