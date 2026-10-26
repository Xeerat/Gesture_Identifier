import logging
import threading
import time

from luma.core.error import DeviceNotFoundError
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306
from PIL import Image

from embedded.patterns import get_pattern
from interfaces import LEDControllerInterface

logger = logging.getLogger(__name__)

# Проверяем наличие Tkinter для возможности эмуляции
try:
    import tkinter as tk

    from PIL import ImageTk

    _HAS_TKINTER = True
except ImportError:
    _HAS_TKINTER = False


class _MockSSD1306:
    """
    Внутренний класс-заглушка, имитирующий интерфейс `luma.oled.device.ssd1306`.

    Позволяет использовать существующий код отрисовки (включая контекстный
    менеджер `luma.core.render.canvas`), но перенаправляет вывод в окно Tkinter
    вместо физической шины I2C. Использует потокобезопасный паттерн: основной
    поток только обновляет переменную под блокировкой, а поток Tkinter опрашивает
    её и отрисовывает.
    """

    def __init__(self, width: int = 128, height: int = 64, scale: int = 4):
        """
        Инициализирует параметры эмулятора и запускает фоновый поток GUI.

        Args:
            width (int): Ширина виртуального дисплея в пикселях.
            height (int): Высота виртуального дисплея в пикселях.
            scale (int): Множитель масштаба для окна предпросмотра.
        """
        self.width = width
        self.height = height
        self.size = (width, height)
        self.mode = "1"
        self._scale = scale
        self._current_image: Image.Image | None = None
        self._lock = threading.Lock()

        # Запускаем GUI в потоке-демоне
        self._thread = threading.Thread(target=self._tkinter_loop, daemon=True)
        self._thread.start()

        # Ждем инициализации root, чтобы избежать race condition
        while not hasattr(self, "root"):
            time.sleep(0.01)

    def _tkinter_loop(self) -> None:
        """
        Целевая функция фонового потока.

        Инициализирует корневое окно Tkinter, настраивает виджеты и запускает
        блокирующий цикл обработки событий `mainloop()`.
        """
        self.root = tk.Tk()
        self.root.title("OLED Mock Preview (SSD1306)")
        self.root.resizable(False, False)

        self.label = tk.Label(self.root, bg="black", bd=0)
        self.label.pack(padx=10, pady=10)

        self.info = tk.Label(
            self.root,
            text="Mock Mode Active (No I2C)",
            fg="gray",
            font=("TkDefaultFont", 9),
        )
        self.info.pack(pady=(0, 10))

        self.photo: ImageTk.PhotoImage | None = None
        self._schedule_update()
        self.root.mainloop()

    def _schedule_update(self) -> None:
        """
        Рекурсивно планирует обновление UI (~15 FPS) через `root.after`.

        Извлекает последнее изображение под блокировкой, масштабирует его
        и конвертирует в режим "RGB" для гарантированно корректного отображения
        в виджете Tkinter Label.
        """
        if not hasattr(self, "root") or not self.root.winfo_exists():
            return

        with self._lock:
            img = self._current_image

        if img is not None:
            # Масштабируем и конвертируем в RGB для надежного отображения в Tkinter
            scaled = img.resize(
                (self.width * self._scale, self.height * self._scale),
                Image.Resampling.NEAREST,
            )
            if scaled.mode != "RGB":
                scaled = scaled.convert("RGB")

            self.photo = ImageTk.PhotoImage(scaled)
            self.label.configure(image=self.photo)

        # Планируем следующий кадр через 66 мс
        self.root.after(66, self._schedule_update)

    def clear(self) -> None:
        """Имитирует очистку дисплея, устанавливая внутренний буфер в черный цвет."""
        with self._lock:
            self._current_image = Image.new(
                "RGB", (self.width, self.height), color="black"
            )

    def display(self, image: Image.Image) -> None:
        """
        Принимает изображение от `luma.canvas` или напрямую и сохраняет его.

        Args:
            image (PIL.Image.Image): Кадр для отображения. Создается копия
                                     изображения для предотвращения мутаций
                                     извне во время отрисовки в UI-потоке.
        """
        with self._lock:
            # Если изображение меньше экрана, центрируем его
            if image.size != (self.width, self.height):
                background = Image.new("1", (self.width, self.height), color=0)
                # Вычисляем смещение для центрирования
                x_offset = (self.width - image.width) // 2
                y_offset = (self.height - image.height) // 2
                background.paste(image, (x_offset, y_offset))
                self._current_image = background
            else:
                self._current_image = image.copy()

    def close(self) -> None:
        """Безопасно инициирует закрытие окна Tkinter из его собственного потока."""
        if hasattr(self, "root") and self.root.winfo_exists():
            self.root.after(0, self.root.destroy)


class OLEDController(LEDControllerInterface):
    """
    Контроллер OLED-дисплея (SSD1306) для отображения жестов и анимаций.

    Реализует интерфейс `LEDControllerInterface`, абстрагируя работу с
    аппаратным обеспечением (I2C) от основного приложения. Использует
    фоновый поток для плавной отрисовки анимации ожидания (RoboEyes)
    без блокировки главного цикла программы.

    При отсутствии физического устройства автоматически переключается в
    режим эмуляции через Tkinter
    """

    def __init__(
        self,
        i2c_port: int = 3,
        i2c_address: int = 0x3C,
        idle_timeout: float = 3.0,
    ):
        """
        Инициализирует контроллер OLED-дисплея SSD1306 и запускает фоновый
        поток для автономной отрисовки кадров и анимации простоя.

        Args:
            i2c_port (int, optional): Номер шины I2C. По умолчанию 3.
            i2c_address (int, optional): HEX-адрес устройства. По умолчанию 0x3C.
            idle_timeout (float, optional): Время бездействия в секундах до
                                            перехода в анимацию ожидания. По умолчанию 3.0.
        Raises:
            DeviceNotFoundError: Если дисплей не найден и эмуляция отключена.
            OSError: Если нет прав доступа к I2C и эмуляция отключена.
            RuntimeError: Если эмуляция запрошена, но библиотека Tkinter не установлена.
        """
        self._lock = threading.Lock()
        self._current_gesture = "roboeyes"
        self._last_update_time = time.time()
        self._idle_timeout = idle_timeout
        self._running = True
        self._is_mock = False

        # Попытка инициализации реального оборудования
        try:
            serial = i2c(port=i2c_port, address=i2c_address)
            self.device = ssd1306(serial, width=128, height=64)
            self.device.clear()
            logger.info(
                f"[OLED] Успешно инициализирован: I2C-{i2c_port}, адрес 0x{i2c_address:02X}"
            )

        except (DeviceNotFoundError, OSError, ImportError) as e:
            # ImportError ловит отсутствие модулей fcntl/smbus2 на Windows
            if _HAS_TKINTER:
                logger.warning(
                    f"[OLED] Аппаратный дисплей недоступен ({type(e).__name__}: {e}). "
                    f"Активирован режим эмуляции (Tkinter)."
                )
                self.device = _MockSSD1306(width=128, height=64, scale=4)
                self.device.clear()
                self._is_mock = True
            else:
                logger.critical(
                    f"[OLED] Критическая ошибка инициализации: {e}. "
                    f"Tkinter недоступен для эмуляции."
                )
                raise

        # Запуск фонового потока отрисовки (daemon=True гарантирует завершение при выходе)
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()

    def show_gesture(self, gesture_name: str) -> None:
        """
        Устанавливает и отображает паттерн, соответствующий распознанному жесту.

        Является неблокирующим вызовом: мгновенно обновляет внутреннее состояние
        под блокировкой, фактическая отрисовка происходит в фоновом потоке.

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
        устройства (реального или mock) перед выполнением команды.
        """
        if self.device:
            self.device.clear()
            logger.debug("[OLED] Экран очищен")

    def set_brightness(self, value: int) -> None:
        """
        Устанавливает уровень яркости (контраста) дисплея.

        Входное значение автоматически ограничивается (clamped) безопасным
        диапазоном от 0 до 255, соответствующим спецификации SSD1306.

        Args:
            value (int): Желаемый уровень яркости (0-255).
        """
        if self.device and not self._is_mock:
            safe_value = max(0, min(255, value))
            self.device.contrast(safe_value)
            logger.info(f"[OLED] Яркость установлена на: {safe_value}")
        elif self._is_mock:
            logger.info(f"[OLED] Запрошена яркость: {value} (игнорируется в эмуляторе)")

    def close(self) -> None:
        """
        Корректно завершает работу контроллера и освобождает аппаратные ресурсы.

        Останавливает фоновый поток отрисовки (с таймаутом ожидания 1.0 сек),
        очищает экран и, в случае использования эмулятора, закрывает окно Tkinter.
        """
        logger.info("[OLED] Завершение работы контроллера...")
        self._running = False

        if self._display_thread.is_alive():
            self._display_thread.join(timeout=1.0)

        if self.device:
            self.device.clear()
            # Если это эмулятор, корректно закрываем его окно
            if self._is_mock and hasattr(self.device, "close"):
                self.device.close()

        logger.info("[OLED] Ресурсы освобождены, поток остановлен.")

    def show_idle(self) -> None:
        """
        Принудительно переводит дисплей в режим ожидания (idle).

        Активирует анимацию "RoboEyes" и сбрасывает таймер последнего обновления,
        чтобы анимация начала воспроизводиться немедленно.
        """
        with self._lock:
            self._current_gesture = "roboeyes"
            self._last_update_time = time.time()
        logger.info("[OLED] Переход в режим ожидания (idle)")

    def _draw_roboeyes_with_blink(self, draw) -> None:
        """
        Отрисовывает кадр анимации ожидания ("RoboEyes") с логикой моргания.

        Моргание происходит циклически: каждые ~2.5 секунды глаза "закрываются"
        (отрисовываются как горизонтальные линии) на 0.15 секунды.

        Args:
            draw: Объект отрисовки `luma.core.render.canvas`, предоставленный
                  контекстным менеджером `canvas`.
        """
        current_time = time.time()
        is_blinking = (current_time % 2.5) < 0.15

        draw.rectangle((0, 0, 127, 63), fill="black")

        if is_blinking:
            draw.line((20, 32, 52, 32), fill="white", width=3)
            draw.line((76, 32, 108, 32), fill="white", width=3)
        else:
            draw.rounded_rectangle((16, 16, 52, 48), radius=6, outline="white", width=2)
            draw.rectangle((28, 26, 40, 38), fill="white")
            draw.rounded_rectangle(
                (76, 16, 112, 48), radius=6, outline="white", width=2
            )
            draw.rectangle((88, 26, 100, 38), fill="white")

    def _display_loop(self) -> None:
        """
        Фоновый цикл отрисовки, работающий в отдельном потоке (~30 FPS).

        На каждом кадре проверяет:
        1. Истек ли `idle_timeout` (если да, переключает состояние на "roboeyes").
        2. Текущий активный жест.
        3. Выполняет либо программную отрисовку анимации через `canvas` (для "roboeyes"),
           либо быструю отрисовку готового 1-битного изображения из HEX-паттерна.
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
                if gesture == "roboeyes":
                    # canvas работает с _MockSSD1306, так как у него есть width, height и метод display()
                    with canvas(self.device) as draw:
                        self._draw_roboeyes_with_blink(draw)
                else:
                    try:
                        hex_data = get_pattern(gesture)
                        # Создаем изображение 64x64 (размер паттерна)
                        image_64x64 = Image.frombytes(
                            "1", (64, 64), bytes.fromhex(hex_data)
                        )

                        # Центрируем на холсте 128x64
                        background = Image.new("1", (128, 64), color=0)
                        x_offset = (128 - 64) // 2  # = 32
                        y_offset = (64 - 64) // 2  # = 0
                        background.paste(image_64x64, (x_offset, y_offset))
                        self.device.display(background)
                    except ValueError:
                        # Если паттерн не найден, рисуем текст об ошибке прямо на холсте
                        with canvas(self.device) as draw:
                            draw.rectangle((0, 0, 127, 63), fill="black")
                            draw.text((10, 25), "UNKNOWN", fill="white")

            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
