import logging

import numpy as np
import onnxruntime as ort

from cv.postprocess import decode_classification_output
from cv.preprocess import preprocess_frame
from interfaces import DetectionResult, DetectorInterface

logger = logging.getLogger(__name__)


class ONNXDetector(DetectorInterface):
    """
    Детектор/классификатор изображений на базе ONNX Runtime (CPU).

    Реализует интерфейс `DetectorInterface` для выполнения инференса
    моделей в формате .onnx. Использует исключительно процессор (CPU)
    через 'CPUExecutionProvider', что делает его идеальным для
    разработки, тестирования на ПК или работы на устройствах без
    специализированных NPU/GPU.
    """

    def __init__(self, config: dict) -> None:
        """
        Инициализирует детектор и загружает ONNX модель в память.

        Args:
            config (dict): Словарь конфигурации. Ожидаемые ключи:
                - "classes" (list[str], optional): Список имен классов.
                  По умолчанию ["none"].
                - "confidence_threshold" (float, optional): Порог уверенности
                  для срабатывания. По умолчанию 0.5.
                - "imgsz" (int, optional): Ожидаемый размер входного изображения
                  (ширина/высота). По умолчанию 224.
                - "onnx_path" (str, optional): Путь к файлу модели .onnx.
                  По умолчанию "src/ml/models/best.onnx".

        Raises:
            FileNotFoundError: Если файл модели по указанному пути не найден.
            ort.capi.onnxruntime_pybind11_state.Fail: Если модель повреждена
                или несовместима с текущей версией ONNX Runtime.
        """
        self._classes = config.get("classes", ["none"])
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

        tensor = preprocess_frame(frame, imgsz=self._imgsz)

        output = self._session.run(None, {self._input_name: tensor})[0]

        gesture, confidence, _ = decode_classification_output(
            output,
            classes=self._classes,
            confidence_threshold=self._threshold,
        )

        return DetectionResult(gesture=gesture, confidence=confidence)

    def close(self) -> None:
        """
        Освобождает ресурсы, связанные с детектором.

        Примечание: ONNX Runtime в Python автоматически управляет памятью
        через сборщик мусора (GC), и явного закрытия сессии не требует.
        Метод реализован в первую очередь для соблюдения контракта
        интерфейса `DetectorInterface` и логирования.
        """
        logger.info("Ресурсы ONNX детектора освобождены.")
        return super().close()

    def warmup(self, frames_count: int = 10) -> None:
        """
        Выполняет "прогрев" (warmup) модели на фиктивных данных.

        Необходим для предварительного выделения памяти и прогрева
        кэша процессора, что обеспечивает стабильное (и минимальное)
        время отклика при первых реальных вызовах метода `detect`.

        Args:
            frames_count (int, optional): Количество фиктивных проходов инференса.
                                          По умолчанию 10. Значения меньше 0
                                          приводятся к 0.
        """
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(max(0, int(frames_count))):
            self.detect(dummy)
        logger.info("Прогрев ONNX детектора успешно выполнен.")
        return super().warmup(frames_count)
