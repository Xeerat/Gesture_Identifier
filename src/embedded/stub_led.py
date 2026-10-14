import logging

from interfaces import LEDControllerInterface

logger = logging.getLogger(__name__)


class StubLEDController(LEDControllerInterface):
    """
    Заглушка (mock) контроллера светодиодной ленты для тестирования пайплайна.

    Имитирует управление hardware, логируя все вызовы методов вместо реальной
    отправки команд на ленту. Используется на этапе разработки, пока реальное
    железо не подключено или отлаживается логика основного приложения.
    """

    def __init__(self) -> None:
        """
        Инициализирует начальное состояние заглушки контроллера.

        Устанавливает яркость по умолчанию (128) и сбрасывает текущий
        отображаемый паттерн в "none".
        """
        self._brightness = 128
        self._current_pattern = "none"

        logger.info("Контроллер LED-ленты успешно инициализирован")

    def show_gesture(self, gesture_name: str) -> None:
        """
        Имитирует отображение паттерна, соответствующего распознанному жесту.

        Избегает избыточного логирования и обновления состояния, если
        запрошенный паттерн совпадает с уже активным.

        Args:
            gesture_name (str): Название жеста (например, "ok", "fist", "none").
        """
        if gesture_name != self._current_pattern:
            logger.info(f"[LED-STUB] Паттерн: {gesture_name}")
            self._current_pattern = gesture_name

    def clear(self) -> None:
        """
        Имитирует выключение всех светодиодов на ленте.

        Сбрасывает внутреннее состояние текущего паттерна в "none".
        """
        logger.info("[LED-STUB] LEDs OFF")
        self._current_pattern = "none"

    def set_brightness(self, value: int) -> None:
        """
        Имитирует установку общей яркости светодиодной ленты.

        Входное значение автоматически ограничивается (clamped)
        безопасным диапазоном от 0 до 255.

        Args:
            value (int): Желаемый уровень яркости.
        """
        self._brightness = max(0, min(255, value))
        logger.info(f"[LED-STUB] Яркость: {self._brightness}")

    def show_idle(self) -> None:
        """
        Имитирует переход ленты в режим ожидания (idle).

        В данной реализации заглушки просто логирует это событие.
        В реальном контроллере здесь может быть анимация или
        тусклое свечение.
        """
        logger.info("[LED-STUB] Паттерн ожидания")

    def close(self) -> None:
        """
        Корректно завершает работу с hardware и освобождает ресурсы.

        Вызывает функцию по умолчанию.
        """
        return super().close()
