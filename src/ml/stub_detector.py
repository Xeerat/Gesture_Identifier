import logging
import random

import numpy as np

from interfaces import DetectionResult, DetectorInterface

logger = logging.getLogger(__name__)


class StubDetector(DetectorInterface):
    """
    Заглушка (mock) детектора жестов для тестирования пайплайна.

    Имитирует работу реальной ML-модели, возвращая случайный жест
    из списка доступных или "none" с заданной вероятностью.
    Используется на этапе разработки, пока реальная модель не подключена.
    """

    def __init__(self, classes: list[str], threshold: float = 0.7) -> None:
        """
        Инициализирует заглушку детектора.

        Args:
            classes (list[str]): Список всех возможных классов жестов.
            threshold (float, optional): Вероятность успешного "распознавания"
                жеста (от 0.0 до 1.0). По умолчанию 0.7.
        """
        self._classes = classes
        self._threshold = threshold
        self._gestures = [c for c in classes if c != "none"]

        logger.info("Детектор успешно инициализирован")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Имитирует распознавание жеста на переданном кадре.

        Игнорирует фактическое содержимое кадра и возвращает случайный результат
        на основе заданного порога вероятности (threshold).

        Args:
            frame (np.ndarray): Кадр изображения в формате BGR.
                Игнорируется заглушкой, но необходим для соответствия
                сигнатуре интерфейса DetectorInterface.

        Returns:
            DetectionResult: Объект с распознанным жестом и уровнем уверенности.
                Если "распознавание" не удалось, возвращает
                gesture="none" и confidence=0.0.
        """
        if random.random() < self._threshold and self._gestures:
            return DetectionResult(
                gesture=random.choice(self._gestures),
                confidence=random.uniform(0.75, 0.99),
            )
        return DetectionResult(gesture="none", confidence=0.0)

    def close(self) -> None:
        """
        Освобождает ресурсы, занятые детектором (память NPU, контекст модели).

        Отправляет сообщение об освобождении ресурсов в лог
        """
        logger.info("Ресурсы детектора освобождены.")
        return super().close()

    def warmup(self, frames_count=10) -> None:
        """
        Выполняет прогрев модели перед началом основной работы.

        Полезно для инициализации NPU или выделения памяти.
        По умолчанию не делает ничего (no-op), переопределяется в реализациях.

        Args:
            frames_count (int, optional): Количество фиктивных кадров для прогрева.
                                          По умолчанию 10.
        """
        logger.info("Прогрев модели детектора успешно выполнен.")
        return super().warmup(frames_count)
