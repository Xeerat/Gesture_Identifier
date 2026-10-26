import logging
from collections import Counter, deque

logger = logging.getLogger(__name__)


class GestureSmoother:
    """
    Модуль сглаживания результатов распознавания жестов.

    Предотвращает "мерцание" (flickering) паттернов на светодиодной ленте,
    используя механизм скользящего окна и мажоритарного голосования.
    Смена отображаемого жеста происходит только при уверенном подтверждении
    модели в течение нескольких кадров подряд.
    """

    def __init__(
        self, buffer_size: int = 7, min_votes: int = 4, none_threshold: int = 5
    ) -> None:
        """
        Инициализирует механизм сглаживания с заданными параметрами чувствительности.

        Args:
            buffer_size (int): Размер скользящего окна (количество последних
                кадров для анализа).
            min_votes (int): Минимальное количество совпадений жеста в окне
                для подтверждения смены состояния.
            none_threshold (int): Количество последовательных "none" в конце
                буфера, необходимое для полного сброса жеста в "none".

        Raises:
            ValueError: Если buffer_size <= 0, min_votes <= 0 или
                min_votes > buffer_size (невозможная логическая конфигурация).
        """
        if buffer_size <= 0 or min_votes <= 0:
            raise ValueError("buffer_size и min_votes должны быть > 0")
        if min_votes > buffer_size:
            raise ValueError("min_votes не может быть больше buffer_size")

        self.buffer_size = buffer_size
        self.min_votes = min_votes
        self.none_threshold = none_threshold

        self._buffer = deque(maxlen=buffer_size)
        self._current_gesture = "none"

        logger.info("Модуль сглаживания успешно инициализирован")

    @property
    def current_gesture(self) -> str:
        """
        Текущий стабилизированный жест.

        Returns:
            str: Название текущего жеста (например, "ok", "fist" или "none"),
                 готовое для передачи в контроллер ленты.
        """
        return self._current_gesture

    def process(self, detected_gesture: str) -> str:
        """
        Обрабатывает новый распознанный жест и возвращает стабилизированное состояние.

        Добавляет новый жест в буфер и применяет логику мажоритарного голосования.
        Если ни один жест не набрал достаточного количества голосов, проверяет,
        не исчез ли жест полностью (анализ последовательных "none" с конца буфера).

        Args:
            detected_gesture (str): Сырой жест, полученный от детектора на текущем кадре.
                Если передано пустое значение или None, интерпретируется как "none".

        Returns:
            str: Стабилизированное название жеста, которое следует использовать
                для обновления состояния периферии.
        """
        gesture = detected_gesture or "none"
        self._buffer.append(gesture)

        counts = Counter(self._buffer)
        most_common_gesture, most_common_votes = counts.most_common(1)[0]

        if most_common_gesture != "none" and most_common_votes >= self.min_votes:
            self._current_gesture = most_common_gesture
        else:
            # Считаем подряд идущие "none" с конца буфера
            consecutive_none = 0
            for g in reversed(self._buffer):
                if g == "none":
                    consecutive_none += 1
                else:
                    break
            if consecutive_none >= self.none_threshold:
                self._current_gesture = "none"

        return self._current_gesture

    def reset(self) -> None:
        """
        Сбрасывает внутреннее состояние сглаживателя.

        Полностью очищает буфер истории и принудительно устанавливает
        текущий жест в "none". Полезно при перезапуске конвейера
        обработки видео или при возникновении критических ошибок детектора.
        """
        self._buffer.clear()
        self._current_gesture = "none"
