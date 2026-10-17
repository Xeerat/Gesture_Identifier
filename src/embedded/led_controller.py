import logging
import threading
import time

from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306
from PIL import Image

from ..interfaces import LEDControllerInterface
from .patterns import get_pattern

logger = logging.getLogger(__name__)


class OLEDController(LEDControllerInterface):
    """
    Контроллер OLED-дисплея (SSD1306) для отображения жестов и анимаций.

    Реализует интерфейс `LEDControllerInterface`, абстрагируя работу с
    аппаратным обеспечением (I2C) от основного приложения. Использует
    фоновый поток для плавной отрисовки анимации ожидания (RoboEyes)
    без блокировки главного цикла программы.
    """

    def __init__(self, i2c_port: int = 3, i2c_address: int = 0x3C, idle_timeout: float = 3.0):
        """
        Инициализирует контроллер дисплея и запускает фоновый поток отрисовки.

        Args:
            i2c_port (int): Номер шины I2C (по умолчанию 3, например, для Raspberry Pi).
            i2c_address (int): I2C-адрес дисплея (по умолчанию 0x3C).
            idle_timeout (float): Время в секундах без обновления жеста,
                                  после которого автоматически включается анимация ожидания.
        """
        self._lock = threading.Lock()
        self._current_gesture = "roboeyes"
        self._last_update_time = time.time()
        self._idle_timeout = idle_timeout
        self._running = True
        
        # Инициализация дисплея
        try:
            serial = i2c(port=i2c_port, address=i2c_address)
            self.device = ssd1306(serial, width=128, height=64)
            self.device.clear()
            logger.info(f"[OLED] Инициализирован: I2C-{i2c_port}, адрес 0x{i2c_address:02X}")
        except Exception as e:
            logger.error(f"[OLED] Ошибка инициализации дисплея: {e}")
            self.device = None
            return

        # Запуск фонового потока отрисовки (daemon=True гарантирует завершение при выходе из программы)
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()

    # --- МЕТОДЫ ИНТЕРФЕЙСА ---

    def show_gesture(self, gesture_name: str) -> None:
        """
        Устанавливает и отображает паттерн, соответствующий распознанному жесту.

        Является неблокирующим вызовом: мгновенно обновляет внутреннее состояние
        и возвращает управление, фактическая отрисовка происходит в фоновом потоке.
        Избегает избыточного обновления состояния и логирования, если запрошенный 
        жест уже является активным.

        Args:
            gesture_name (str): Название жеста (например, "ok", "fist", "none").
        """
        normalized_gesture = gesture_name.lower().strip()
        with self._lock:
            if normalized_gesture != self._current_gesture:
                self._current_gesture = normalized_gesture
                self._last_update_time = time.time()
                logger.info(f"[OLED] Отображение жеста: {normalized_gesture}")

    def clear(self) -> None:
        """
        Очищает экран дисплея, выключая все активные пиксели.

        Безопасный вызов: предварительно проверяет наличие инициализированного 
        устройства перед выполнением аппаратной команды.
        """
        if self.device:
            self.device.clear()
            logger.debug("[OLED] Экран очищен")

    def set_brightness(self, value: int) -> None:
        """
        Устанавливает уровень яркости (контраста) дисплея.

        Входное значение автоматически ограничивается (clamped)
        безопасным диапазоном от 0 до 255, соответствующим спецификации SSD1306.

        Args:
            value (int): Желаемый уровень яркости (0-255).
        """
        if self.device:
            safe_value = max(0, min(255, value))
            self.device.contrast(safe_value)
            logger.info(f"[OLED] Яркость установлена на: {safe_value}")

    def close(self) -> None:
        """
        Корректно завершает работу контроллера и освобождает аппаратные ресурсы.

        Останавливает фоновый поток отрисовки (с таймаутом ожидания 1.0 сек)
        и очищает экран дисплея перед завершением. Рекомендуется вызывать
        при штатном завершении работы основного приложения.
        """
        logger.info("[OLED] Завершение работы контроллера...")
        self._running = False
        if self._display_thread.is_alive():
            self._display_thread.join(timeout=1.0)
        if self.device:
            self.device.clear()
        logger.info("[OLED] Ресурсы освобождены, поток остановлен.")

    def show_idle(self) -> None:
        """
        Принудительно переводит дисплей в режим ожидания (idle).

        Активирует анимацию "RoboEyes" и сбрасывает таймер последнего
        обновления, чтобы анимация начала воспроизводиться немедленно,
        не дожидаясь истечения `idle_timeout`.
        """
        with self._lock:
            self._current_gesture = "roboeyes"
            self._last_update_time = time.time()
        logger.info("[OLED] Переход в режим ожидания (idle)")

    # --- ВНУТРЕННЯЯ ЛОГИКА ---

    def _draw_roboeyes_with_blink(self, draw) -> None:
        """
        Отрисовывает кадр анимации ожидания ("RoboEyes") с логикой моргания.

        Моргание происходит циклически: каждые ~2.5 секунды глаза
        "закрываются" (отрисовываются как горизонтальные линии) на 0.15 секунды.

        Args:
            draw: Объект отрисовки `luma.core.render.canvas`,
                  предоставленный контекстным менеджером `canvas`.
        """
        current_time = time.time()
        # Моргание каждые ~2.5 секунды на 0.15 секунды
        is_blinking = (current_time % 2.5) < 0.15 
        
        draw.rectangle((0, 0, 127, 63), fill="black") # Очистка фона
        
        if is_blinking:
            # Закрытые глаза (две линии)
            draw.line((20, 32, 52, 32), fill="white", width=3)
            draw.line((76, 32, 108, 32), fill="white", width=3)
        else:
            # Открытые глаза (стиль FluxGarage)
            # Левый глаз
            draw.rounded_rectangle((16, 16, 52, 48), radius=6, outline="white", width=2)
            draw.rectangle((28, 26, 40, 38), fill="white") # Зрачок
            # Правый глаз
            draw.rounded_rectangle((76, 16, 112, 48), radius=6, outline="white", width=2)
            draw.rectangle((88, 26, 100, 38), fill="white") # Зрачок

    def _display_loop(self) -> None:
        """
        Фоновый цикл отрисовки, работающий в отдельном потоке.

        Обеспечивает частоту обновления ~30 FPS, предотвращая 100% загрузку CPU.
        На каждом кадре проверяет:
        1. Истек ли `idle_timeout` (если да, переключает состояние на "roboeyes").
        2. Текущий активный жест.
        3. Выполняет либо программную отрисовку анимации (для "roboeyes"),
           либо быструю отрисовку готового изображения из HEX-паттерна.
        """
        frame_time = 1.0 / 30.0 
        
        while self._running:
            loop_start = time.time()
            
            with self._lock:
                gesture = self._current_gesture
                time_since_update = time.time() - self._last_update_time

            if time_since_update > self._idle_timeout and gesture != "roboeyes":
                gesture = "roboeyes"

            if self.device:
                # Ожидание
                if gesture == "roboeyes":
                    with canvas(self.device) as draw:
                        self._draw_roboeyes_with_blink(draw)
                # Паттерны жестов
                else:
                    hex_data = get_pattern(gesture)
                    image = Image.frombytes('1', (128, 64), bytes.fromhex(hex_data))
                    self.device.display(image)

            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)