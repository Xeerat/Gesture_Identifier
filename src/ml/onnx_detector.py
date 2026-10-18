"""
Детектор жестов через ONNX Runtime (режим ПК).
Реализует DetectorInterface из interfaces.py.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnxruntime as ort

from interfaces import DetectionResult, DetectorInterface
from cv.preprocess import preprocess_frame
from cv.postprocess import decode_yolov8_output, rescale_boxes

logger = logging.getLogger(__name__)

FULL_DIAG_FRAMES = 10


class ONNXDetector(DetectorInterface):
    """Детектор жестов на базе ONNX Runtime (CPU, для ноутбука)."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.threshold = config.get("confidence_threshold", 0.7)
        self.iou_thresh = config.get("iou_threshold", 0.45)
        self.classes = config.get("classes", [])

        # --- нормализация входа ---
        # "none"  -> пиксели как float32 в диапазоне [0, 255] (нормализация внутри графа)
        # "255"   -> пиксели / 255.0, диапазон [0, 1] (стандартный ultralytics экспорт)
        self.normalization = config.get("input_normalization", "none")

        # --- цветовой канал ---
        # "rgb" -> предобработка Маши (BGR->RGB)
        # "bgr" -> без конвертации цвета
        self.color_space = config.get("input_color", "rgb")

        # --- маппинг классов ---
        self.label_map: Dict[str, str] = {}
        labels_path = config.get("labels_path")

        if labels_path and Path(labels_path).exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                self.label_map = json.load(f)
            logger.info("[DETECTOR] labels из %s (%d классов)", labels_path, len(self.label_map))
        elif self.classes:
            self.label_map = {str(i): name for i, name in enumerate(self.classes)}
            logger.info("[DETECTOR] labels из YAML classes (%d классов)", len(self.label_map))
        else:
            logger.warning("[DETECTOR] ВНИМАНИЕ: ни labels_path, ни classes не заданы!")

        logger.info("[DETECTOR] label_map = %s", self.label_map)

        # --- загрузка модели ---
        model_path = config.get("onnx_path", "src/ml/models/gesture_yolov8n.onnx")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX модель не найдена: {model_path}")

        logger.info("[DETECTOR] Загрузка ONNX модели: %s", model_path)
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # --- мета модели ---
        input_meta = self.session.get_inputs()[0]
        output_meta = self.session.get_outputs()[0]

        logger.info("[DETECTOR] === МЕТА МОДЕЛИ ===")
        logger.info("[DETECTOR] INPUT : name=%s shape=%s type=%s",
                    input_meta.name, input_meta.shape, input_meta.type)
        logger.info("[DETECTOR] OUTPUT: name=%s shape=%s type=%s",
                    output_meta.name, output_meta.shape, output_meta.type)

        # Определяем layout
        self._input_is_nchw = False
        if len(input_meta.shape) == 4 and str(input_meta.shape[1]) == "3":
            self._input_is_nchw = True

        logger.info("[DETECTOR] NCHW=%s normalization=%s color=%s",
                    self._input_is_nchw, self.normalization, self.color_space)

        self._frame_count = 0

    def _build_input(self, frame: np.ndarray) -> tuple:
        """Предобработка + формирование входного тензора."""
        if self.color_space == "bgr":
            # Пропускаем BGR->RGB конвертацию Маши, делаем свой letterbox
            from cv.preprocess import letterbox
            img, ratio, pad = letterbox(frame, new_shape=(640, 640))
            # img остаётся BGR
        else:
            img_rgb, ratio, pad = preprocess_frame(frame)
            img = img_rgb  # RGB

        input_tensor = np.expand_dims(img, axis=0)  # [1, 640, 640, 3]

        # Нормализация
        if self.normalization == "none":
            # Пиксели остаются 0-255, просто cast в float32
            input_tensor = input_tensor.astype(np.float32)
        elif self.normalization == "255":
            input_tensor = input_tensor.astype(np.float32) / 255.0
        else:
            input_tensor = input_tensor.astype(np.float32)

        # Transpose в NCHW если нужно
        if self._input_is_nchw and input_tensor.ndim == 4 and input_tensor.shape[-1] == 3:
            input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))

        return input_tensor, ratio, pad

    def detect(self, frame: np.ndarray) -> DetectionResult:
        self._frame_count += 1
        full = self._frame_count <= FULL_DIAG_FRAMES
        orig_shape = frame.shape[:2]

        if full:
            logger.info("[DETECTOR] ========== FRAME %d ==========", self._frame_count)

        # 1. Предобработка + входной тензор
        input_tensor, ratio, pad = self._build_input(frame)

        if full:
            logger.info("[DETECTOR] INPUT: shape=%s dtype=%s min=%.2f max=%.2f",
                        input_tensor.shape, input_tensor.dtype,
                        float(input_tensor.min()), float(input_tensor.max()))

        # 2. Инференс
        raw_output = self.session.run(None, {self.input_name: input_tensor})[0]

        if full:
            logger.info("[DETECTOR] OUTPUT: shape=%s min=%.4f max=%.4f",
                        raw_output.shape,
                        float(raw_output.min()), float(raw_output.max()))

            # TOP5 диагностика
            diag = np.squeeze(raw_output)
            if diag.ndim == 2:
                if diag.shape[0] < diag.shape[1]:
                    diag = diag.T
                if diag.shape[1] > 4:
                    scores = diag[:, 4:].astype(np.float32)
                    if float(scores.min()) < 0.0 or float(scores.max()) > 1.0:
                        scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -50, 50)))
                        logger.info("[DETECTOR] TOP5: sigmoid применён")
                    max_scores = scores.max(axis=1)
                    class_ids = scores.argmax(axis=1)
                    top_idx = np.argsort(max_scores)[-5:][::-1]
                    top_list = [
                        (int(class_ids[i]),
                         self.label_map.get(str(int(class_ids[i])), "?"),
                         round(float(max_scores[i]), 4))
                        for i in top_idx
                    ]
                    logger.info("[DETECTOR] TOP5: %s", top_list)

        # 3. Постобработка
        boxes = decode_yolov8_output(
            raw_output,
            conf_thresh=self.threshold,
            iou_thresh=self.iou_thresh,
        )

        # 4. Выбор лучшей детекции
        if len(boxes) > 0:
            boxes = rescale_boxes(boxes, ratio, pad, orig_shape)
            best_idx = int(np.argmax(boxes[:, 4]))
            best_box = boxes[best_idx]

            confidence = float(best_box[4])
            class_id = int(best_box[5])
            gesture = self.label_map.get(str(class_id), "none")
            bbox = (int(best_box[0]), int(best_box[1]),
                    int(best_box[2]), int(best_box[3]))

            logger.info("[DETECTOR] MODEL_RAW: id=%d label=%s conf=%.4f bbox=%s",
                        class_id, gesture, confidence, bbox)

            return DetectionResult(gesture=gesture, confidence=confidence, bbox=bbox)

        if full:
            logger.info("[DETECTOR] MODEL_RAW: no_detection")
        return DetectionResult(gesture="none", confidence=0.0)

    def warmup(self, frames_count: int = 10) -> None:
        logger.info("[DETECTOR] Прогрев (%d кадров)...", frames_count)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(frames_count):
            self.detect(dummy)
        self._frame_count = 0
        logger.info("[DETECTOR] Прогрев завершён.")

    def close(self) -> None:
        logger.info("[DETECTOR] ONNX детектор остановлен.")