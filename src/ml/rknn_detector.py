import logging
from pathlib import Path
import numpy as np

from interfaces import DetectorInterface, DetectionResult
from cv.preprocess import preprocess_frame
from cv.postprocess import decode_yolov8_output, rescale_boxes

logger = logging.getLogger(__name__)


class RKNN_detector(DetectorInterface):
    """Детектор жестов через RKNN (режим платы Firefly)."""

    def __init__(self, config: dict) -> None:
        from rknnlite.api import RKNNLite

        self.classes = config.get("classes", ["none"])
        self.threshold = config.get("confidence_threshold", 0.7)
        self.iou_thresh = config.get("iou_threshold", 0.45)

        model_path = config.get("rknn_path", "models/gesture_yolov8n.rknn")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"RKNN модель не найдена: {model_path}")

        logger.info(f"Загрузка RKNN модели: {model_path}")
        self.rknn = RKNNLite()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime()

    def detect(self, frame: np.ndarray) -> DetectionResult:
        orig_shape = frame.shape[:2]

        img_rgb, ratio, pad = preprocess_frame(frame)
        input_tensor = np.expand_dims(img_rgb, axis=0)

        raw_output = self.rknn.inference(inputs=[input_tensor])[0]
        boxes = decode_yolov8_output(
            raw_output, conf_thresh=self.threshold, iou_thresh=self.iou_thresh
        )

        if len(boxes) > 0:
            boxes = rescale_boxes(boxes, ratio, pad, orig_shape)
            best_idx = int(np.argmax(boxes[:, 4]))
            best_box = boxes[best_idx]

            confidence = float(best_box[4])
            class_id = int(best_box[5])
            gesture = (
                self.classes[class_id] if class_id < len(self.classes) else "none"
            )
            bbox = (
                int(best_box[0]),
                int(best_box[1]),
                int(best_box[2]),
                int(best_box[3]),
            )

            return DetectionResult(
                gesture=gesture, confidence=confidence, bbox=bbox
            )

        return DetectionResult(gesture="none", confidence=0.0)

    def warmup(self, frames_count: int = 10) -> None:
        logger.info(f"Прогрев NPU ({frames_count} кадров)...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(frames_count):
            self.detect(dummy)

    def close(self) -> None:
        if hasattr(self, "rknn"):
            self.rknn.release()
        logger.info("RKNN детектор остановлен.")