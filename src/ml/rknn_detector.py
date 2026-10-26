import logging

import numpy as np
from rknnlite.api import RKNNLite

from cv.postprocess import decode_classification_output
from cv.preprocess import preprocess_frame_rknn
from interfaces import DetectionResult, DetectorInterface

logger = logging.getLogger(__name__)


class RKNN_detector(DetectorInterface):
    """
    Детектор/классификатор изображений на базе NPU Rockchip (RKNN).

    Реализует интерфейс `DetectorInterface` для выполнения инференса
    предварительно сконвертированных моделей формата .rknn. Обеспечивает
    загрузку модели, предобработку кадров, выполнение инференса и
    постобработку результатов.
    """

    def __init__(self, config: dict) -> None:
        """
        Инициализирует RKNN детектор и загружает модель в память NPU.

        Args:
            config (dict): Словарь конфигурации. Ожидаемые ключи:
                - "classes" (list[str]): Список имен классов для классификации.
                - "confidence_threshold" (float, optional): Порог уверенности для
                  срабатывания детекции. По умолчанию 0.5.
                - "imgsz" (int, optional): Ожидаемый размер входного изображения
                  (ширина/высота). По умолчанию 224.
                - "rknn_path" (str, optional): Путь к файлу модели .rknn.
                  По умолчанию "src/ml/models/best.rknn".

        Raises:
            FileNotFoundError: Если файл модели по указанному пути не найден
                               или не может быть загружен.
            RuntimeError: Если не удалось инициализировать RKNN runtime на устройстве
                          (например, отсутствуют драйверы NPU или права доступа).
        """
        self._classes = config.get("classes", ["none"])
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
        """
        Выполняет предсказание (инференс) для одного входного кадра.

        Args:
            frame (np.ndarray | None): Входное изображение в формате numpy array
                                       (обычно shape HxWxC, BGR или RGB).
                                       Допускается передача None.

        Returns:
            DetectionResult: Объект с результатом детекции, содержащий:
                - gesture (str): Название предсказанного класса. Возвращает "none",
                                 если кадр пустой или уверенность ниже порога.
                - confidence (float): Уверенность модели в предсказании (от 0.0 до 1.0).
        """
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
        """
        Освобождает ресурсы NPU и закрывает сессию RKNN.

        Рекомендуется явно вызывать этот метод при завершении работы приложения
        или при динамическом удалении экземпляра детектора для предотвращения
        утечек памяти и блокировки устройства NPU.
        """
        if hasattr(self, "_rknn"):
            self._rknn.release()
        logger.info("Ресурсы RKNN детектора освобождены.")
        return super().close()

    def warmup(self, frames_count: int = 10) -> None:
        """
        Выполняет "прогрев" (warmup) модели на фиктивных данных.

        Необходим для инициализации внутренних буферов памяти NPU и
        достижения стабильного (максимального) времени инференса при
        первых реальных вызовах метода `detect`.

        Args:
            frames_count (int, optional): Количество фиктивных проходов инференса.
                                          По умолчанию 10. Значения меньше 0
                                          приводятся к 0.
        """
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(max(0, int(frames_count))):
            self.detect(dummy)
        logger.info("Прогрев RKNN детектора успешно выполнен.")
        return super().warmup(frames_count)
