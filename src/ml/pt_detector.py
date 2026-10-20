"""
Детектор жестов через PyTorch classification-модель best.pt.

Используется как эталонная проверка пайплайна:
камера -> main.py -> PTDetector -> ultralytics -> DetectionResult -> smoothing -> LED.

Здесь Ultralytics сам делает предобработку и постобработку.
Модель classification, поэтому боксов нет.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from ultralytics import YOLO

from interfaces import DetectionResult, DetectorInterface

logger = logging.getLogger(__name__)


class PTDetector(DetectorInterface):
    """
    Детектор жестов на базе PyTorch classification-модели.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Инициализирует PT-детектор.

        Args:
            config (Dict[str, Any]): Секция model из конфига.
        """
        self._classes: List[str] = list(config.get("classes", []))
        self._threshold = float(config.get("confidence_threshold", 0.5))
        self._imgsz = int(config.get("imgsz", 224))

        model_path = config.get("pt_path", "src/ml/models/best.pt")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"PT модель не найдена: {model_path}")

        self._model = YOLO(str(model_path))

        logger.info("PT детектор успешно инициализирован")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Распознаёт жест на переданном кадре.

        Args:
            frame (np.ndarray): Кадр изображения в формате BGR.

        Returns:
            DetectionResult: Объект с распознанным жестом и уровнем уверенности.
                Если жест не распознан, возвращает
                gesture="none" и confidence=0.0.
        """
        if frame is None:
            return DetectionResult(gesture="none", confidence=0.0)

        results = self._model.predict(
            source=frame,
            imgsz=self._imgsz,
            verbose=False,
            device="cpu",
        )

        if not results:
            return DetectionResult(gesture="none", confidence=0.0)

        probs = getattr(results[0], "probs", None)

        if probs is None:
            return DetectionResult(gesture="none", confidence=0.0)

        class_id = int(getattr(probs, "top1", -1))
        confidence = float(getattr(probs, "top1conf", 0.0))

        if (
            class_id < 0
            or class_id >= len(self._classes)
            or confidence < self._threshold
        ):
            return DetectionResult(gesture="none", confidence=0.0)

        gesture = self._classes[class_id]

        return DetectionResult(
            gesture=gesture,
            confidence=confidence,
        )

    def close(self) -> None:
        """
        Освобождает ресурсы детектора.
        """
        logger.info("Ресурсы PT детектора освобождены.")
        return super().close()

    def warmup(self, frames_count: int = 10) -> None:
        """
        Выполняет прогрев модели перед началом основной работы.

        Args:
            frames_count (int, optional): Количество фиктивных кадров для прогрева.
                                          По умолчанию 10.
        """
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)

        for _ in range(max(0, int(frames_count))):
            self.detect(dummy)

        logger.info("Прогрев PT детектора успешно выполнен.")
        return super().warmup(frames_count)