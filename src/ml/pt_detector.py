"""
Детектор жестов через оригинальную PyTorch-модель best.pt + ultralytics.

Этот вариант используется для диагностики полного пайплайна:
камера -> main.py -> PTDetector -> ultralytics -> DetectionResult -> smoothing -> LED.

Здесь мы НЕ используем наши cv.preprocess/cv.postprocess,
потому что ultralytics сам делает предобработку и постобработку.
Это позволяет проверить: живая ли модель вообще и приходит ли нормальный кадр с камеры.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from interfaces import DetectionResult, DetectorInterface

logger = logging.getLogger(__name__)


class PTDetector(DetectorInterface):
    """
    Детектор на базе best.pt через ultralytics.
    Максимальная диагностика для проверки камеры, модели и вывода.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        from ultralytics import YOLO

        self.threshold = float(config.get("confidence_threshold", 0.001))
        self.iou_thresh = float(config.get("iou_threshold", 0.99))
        self.imgsz = int(config.get("imgsz", 640))
        self.input_color = str(config.get("input_color", "bgr")).lower()

        self.save_debug_frames = bool(config.get("save_debug_frames", True))
        self.save_debug_every = max(1, int(config.get("save_debug_every", 1)))
        self.max_debug_frames = int(config.get("max_debug_frames", 60))
        self.debug_dir = Path(config.get("debug_dir", "debug_pt"))

        if self.save_debug_frames:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info("[PT] Debug-папка: %s", self.debug_dir.resolve())

        model_path = config.get("pt_path", "src/ml/models/gesture_yolov8n.pt")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"PT модель не найдена: {model_path}")

        logger.info("[PT] Загрузка модели: %s", model_path)
        self.model = YOLO(str(model_path))

        raw_names = getattr(self.model, "names", None)
        if raw_names is None:
            raw_names = getattr(self.model.model, "names", {})

        self.label_map: Dict[int, str] = {
            int(k): str(v) for k, v in raw_names.items()
        }

        logger.info("[PT] === МЕТА МОДЕЛИ ===")
        logger.info("[PT] task=%s", getattr(self.model, "task", "unknown"))
        logger.info("[PT] классов в модели: %d", len(self.label_map))
        logger.info("[PT] label_map: %s", self.label_map)
        logger.info(
            "[PT] conf_thresh=%.6f iou_thresh=%.6f imgsz=%d input_color=%s",
            self.threshold,
            self.iou_thresh,
            self.imgsz,
            self.input_color,
        )

        self._frame_count = 0

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Приводим кадр к виду, который ожидает ultralytics:
        BGR, uint8, контiguous.
        """
        if frame is None:
            raise ValueError("[PT] frame is None")

        if frame.dtype != np.uint8:
            logger.warning(
                "[PT] frame dtype не uint8: %s. Привожу к uint8.",
                frame.dtype,
            )
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if frame.ndim == 2:
            logger.warning("[PT] frame grayscale. Конвертирую в BGR.")
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if frame.ndim != 3:
            raise ValueError(f"[PT] Неожиданная размерность frame: {frame.shape}")

        if frame.shape[2] == 4:
            logger.warning("[PT] frame BGRA. Конвертирую в BGR.")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if frame.shape[2] != 3:
            raise ValueError(f"[PT] Неожиданное число каналов: {frame.shape}")

        # По умолчанию считаем, что OpenCV уже дал BGR.
        # Если в main.py почему-то отдаёт RGB, можно поставить input_color: "rgb".
        if self.input_color == "rgb":
            logger.debug("[PT] Конвертирую RGB -> BGR для ultralytics")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif self.input_color != "bgr":
            logger.warning(
                "[PT] Неизвестный input_color=%s. Оставляю как есть.",
                self.input_color,
            )

        return np.ascontiguousarray(frame)

    def _frame_stats(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Простая статистика кадра, чтобы понять:
        - чёрный ли экран;
        - однотонный ли;
        - есть ли вообще перепады яркости.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        std = float(gray.std())
        min_v = int(gray.min())
        max_v = int(gray.max())

        # Примерная оценка "не однотонности"
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)

        return {
            "shape": frame.shape,
            "dtype": str(frame.dtype),
            "mean": round(mean, 3),
            "std": round(std, 3),
            "min": min_v,
            "max": max_v,
            "edge_ratio": round(edge_ratio, 5),
        }

    def _should_save_debug(self, frame_idx: int) -> bool:
        if not self.save_debug_frames:
            return False
        if frame_idx > self.max_debug_frames:
            return False
        return frame_idx == 1 or frame_idx % self.save_debug_every == 0

    def _save_raw_frame(self, frame: np.ndarray, frame_idx: int) -> Path:
        path = self.debug_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(path), frame)
        return path

    def _draw_detections(
        self,
        frame: np.ndarray,
        boxes_xyxy: np.ndarray,
        confs: np.ndarray,
        cls_ids: np.ndarray,
    ) -> np.ndarray:
        annotated = frame.copy()

        for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confs, cls_ids):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cls_id = int(cls_id)
            conf = float(conf)
            label = self.label_map.get(cls_id, f"class_{cls_id}")

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"{label} {conf:.3f}"
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            cv2.rectangle(
                annotated,
                (x1, y1 - th - baseline - 6),
                (x1 + tw, y1),
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                annotated,
                text,
                (x1, y1 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        return annotated

    def _save_annotated_frame(
        self,
        frame: np.ndarray,
        boxes_xyxy: np.ndarray,
        confs: np.ndarray,
        cls_ids: np.ndarray,
        frame_idx: int,
    ) -> Path:
        annotated = self._draw_detections(frame, boxes_xyxy, confs, cls_ids)
        path = self.debug_dir / f"annotated_{frame_idx:06d}.jpg"
        cv2.imwrite(str(path), annotated)
        return path

    def _save_no_detection_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ) -> Path:
        annotated = frame.copy()

        cv2.putText(
            annotated,
            "NO DETECTION",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        path = self.debug_dir / f"no_det_{frame_idx:06d}.jpg"
        cv2.imwrite(str(path), annotated)
        return path

    def detect(self, frame: np.ndarray) -> DetectionResult:
        self._frame_count += 1
        frame_idx = self._frame_count

        full_log = frame_idx <= 30

        try:
            model_frame = self._prepare_frame(frame)
        except Exception as e:
            logger.exception("[PT] Ошибка подготовки кадра: %s", e)
            return DetectionResult(gesture="none", confidence=0.0)

        stats = self._frame_stats(model_frame)

        if full_log:
            logger.info("[PT] ========== FRAME %d ==========", frame_idx)
            logger.info("[PT] FRAME STATS: %s", stats)

        # Сохраняем сырой кадр, который реально уходит в модель
        if self._should_save_debug(frame_idx):
            raw_path = self._save_raw_frame(model_frame, frame_idx)
            if full_log:
                logger.info("[PT] Сохранён raw кадр: %s", raw_path)

        # Запуск ultralytics
        try:
            results = self.model.predict(
                source=model_frame,
                conf=self.threshold,
                iou=self.iou_thresh,
                imgsz=self.imgsz,
                max_det=1000,
                verbose=False,
                device="cpu",
            )
        except Exception as e:
            logger.exception("[PT] Ошибка model.predict: %s", e)
            return DetectionResult(gesture="none", confidence=0.0)

        if not results:
            logger.warning("[PT] results пустой")
            if self._should_save_debug(frame_idx):
                path = self._save_no_detection_frame(model_frame, frame_idx)
                logger.info("[PT] Сохранён no_det кадр: %s", path)
            return DetectionResult(gesture="none", confidence=0.0)

        result = results[0]
        boxes = getattr(result, "boxes", None)

        if boxes is None:
            if full_log:
                logger.info("[PT] boxes=None -> no_detection")
            if self._should_save_debug(frame_idx):
                path = self._save_no_detection_frame(model_frame, frame_idx)
                if full_log:
                    logger.info("[PT] Сохранён no_det кадр: %s", path)
            return DetectionResult(gesture="none", confidence=0.0)

        if len(boxes) == 0:
            if full_log:
                logger.info("[PT] boxes len=0 -> no_detection")
            if self._should_save_debug(frame_idx):
                path = self._save_no_detection_frame(model_frame, frame_idx)
                if full_log:
                    logger.info("[PT] Сохранён no_det кадр: %s", path)
            return DetectionResult(gesture="none", confidence=0.0)

        # ultralytics уже отдаёт координаты в исходном кадре
        confs = boxes.conf.cpu().numpy().astype(np.float32)
        cls_ids = boxes.cls.cpu().numpy().astype(np.int64)
        boxes_xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)

        if full_log:
            detections: List[Tuple[int, str, float, Tuple[int, int, int, int]]] = []
            for conf, cls_id, box in zip(confs, cls_ids, boxes_xyxy):
                cls_id = int(cls_id)
                conf = float(conf)
                x1, y1, x2, y2 = map(int, box)
                label = self.label_map.get(cls_id, "unknown")
                detections.append((cls_id, label, conf, (x1, y1, x2, y2)))

            logger.info("[PT] Всего детекций: %d", len(detections))
            logger.info("[PT] Все детекции: %s", detections)

            dist: Dict[int, int] = {}
            for cls_id, _, _, _ in detections:
                dist[cls_id] = dist.get(cls_id, 0) + 1
            logger.info("[PT] Распределение class_id: %s", dist)

        # Сохраняем аннотированный кадр, если есть детекции
        if self._should_save_debug(frame_idx):
            ann_path = self._save_annotated_frame(
                model_frame,
                boxes_xyxy,
                confs,
                cls_ids,
                frame_idx,
            )
            if full_log:
                logger.info("[PT] Сохранён annotated кадр: %s", ann_path)

        best_idx = int(np.argmax(confs))
        confidence = float(confs[best_idx])
        class_id = int(cls_ids[best_idx])
        x1, y1, x2, y2 = map(int, boxes_xyxy[best_idx])

        gesture = self.label_map.get(class_id, "none")

        logger.info(
            "[PT] MODEL_RAW: id=%d label=%s conf=%.6f bbox=(%d,%d,%d,%d)",
            class_id,
            gesture,
            confidence,
            x1,
            y1,
            x2,
            y2,
        )

        return DetectionResult(
            gesture=gesture,
            confidence=confidence,
            bbox=(x1, y1, x2, y2),
        )

    def warmup(self, frames_count: int = 5) -> None:
        logger.info("[PT] Прогрев (%d кадров)...", frames_count)

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)

        for _ in range(frames_count):
            try:
                self.model.predict(
                    source=dummy,
                    conf=0.001,
                    iou=0.99,
                    imgsz=self.imgsz,
                    max_det=1000,
                    verbose=False,
                    device="cpu",
                )
            except Exception as e:
                logger.warning("[PT] Ошибка при прогреве: %s", e)

        self._frame_count = 0
        logger.info("[PT] Прогрев завершён, счётчик кадров сброшен.")

    def close(self) -> None:
        logger.info("[PT] Детектор остановлен.")