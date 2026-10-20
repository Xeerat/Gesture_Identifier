"""
RKNN-детектор жестов для classification-модели (плата rk3588).
Вход: float32 NCHW [0,255]; /255 встроено в .rknn через std_values=255.
Выход модели: вероятности [1,7] (softmax уже в графе).
"""

import logging
from typing import Any, Dict, List

import numpy as np

from cv.preprocess import preprocess_frame_rknn
from cv.postprocess import decode_classification_output
from interfaces import DetectionResult, DetectorInterface

logger = logging.getLogger(__name__)


class RKNN_detector(DetectorInterface):

    def __init__(self, config: Dict[str, Any]) -> None:
        from rknnlite.api import RKNNLite

        self._classes: List[str] = list(config.get("classes", []))
        self._threshold = float(config.get("confidence_threshold", 0.5))
        self._imgsz = int(config.get("imgsz", 224))

        model_path = config.get("rknn_path", "src/ml/models/best.rknn")

        self._rknn = RKNNLite()
        if self._rknn.load_rknn(model_path) != 0:
            raise FileNotFoundError(f"RKNN модель не найдена: {model_path}")
        if self._rknn.init_runtime() != 0:
            raise RuntimeError("Не удалось инициализировать RKNN runtime")

        logger.info("RKNN детектор успешно инициализирован")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if frame is None:
            return DetectionResult(gesture="none", confidence=0.0)

        tensor = preprocess_frame_rknn(frame, imgsz=self._imgsz)

        outputs = self._rknn.inference(inputs=[tensor])
        if not outputs:
            return DetectionResult(gesture="none", confidence=0.0)

        probs = np.asarray(outputs[0], dtype=np.float32)

        gesture, confidence, _ = decode_classification_output(
            probs,
            classes=self._classes,
            confidence_threshold=self._threshold,
        )

        return DetectionResult(gesture=gesture, confidence=confidence)

    def close(self) -> None:
        if hasattr(self, "_rknn"):
            self._rknn.release()
        logger.info("Ресурсы RKNN детектора освобождены.")
        return super().close()

    def warmup(self, frames_count: int = 10) -> None:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(max(0, int(frames_count))):
            self.detect(dummy)
        logger.info("Прогрев RKNN детектора успешно выполнен.")
        return super().warmup(frames_count)