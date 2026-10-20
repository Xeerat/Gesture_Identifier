"""
ONNX-детектор жестов для classification-модели.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort

from cv.preprocess import preprocess_frame
from cv.postprocess import decode_classification_output
from interfaces import DetectionResult, DetectorInterface

logger = logging.getLogger(__name__)


class ONNXDetector(DetectorInterface):

    def __init__(self, config: Dict[str, Any]) -> None:
        self._classes: List[str] = list(config.get("classes", []))
        self._threshold = float(config.get("confidence_threshold", 0.5))
        self._imgsz = int(config.get("imgsz", 224))

        model_path = config.get("onnx_path", "src/ml/models/best.onnx")

        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

        logger.info("ONNX детектор успешно инициализирован")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if frame is None:
            return DetectionResult(gesture="none", confidence=0.0)

        tensor = preprocess_frame(frame, imgsz=self._imgsz)

        output = self._session.run(None, {self._input_name: tensor})[0]

        gesture, confidence, _ = decode_classification_output(
            output,
            classes=self._classes,
            confidence_threshold=self._threshold,
        )

        return DetectionResult(gesture=gesture, confidence=confidence)

    def close(self) -> None:
        logger.info("Ресурсы ONNX детектора освобождены.")
        return super().close()

    def warmup(self, frames_count: int = 10) -> None:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(max(0, int(frames_count))):
            self.detect(dummy)
        logger.info("Прогрев ONNX детектора успешно выполнен.")
        return super().warmup(frames_count)