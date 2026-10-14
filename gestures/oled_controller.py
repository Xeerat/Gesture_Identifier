import time
import threading

from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306
from PIL import Image

from .gesture_patterns.py import get_pattern


class OLEDController:
    def __init__(self, i2c_port=3, i2c_address=0x3C, idle_timeout=3.0):
        """
        :param idle_timeout: Секунды без нового жеста, после которых включаются RoboEyes.
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
            print(f"[OLED] Инициализирован: I2C-{i2c_port}, адрес 0x{i2c_address:02X}")
        except Exception as e:
            print(f"[OLED] Ошибка инициализации: {e}")
            self.device = None
            return

        # Запуск фонового потока отрисовки (daemon=True гарантирует завершение при выходе из программы)
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()

    def show_gesture(self, gesture_name: str):
        """
        НЕБЛОКИРУЮЩИЙ вызов. Мгновенно обновляет состояние и возвращает управление.
        Идеально для главного цикла Тимлида.
        """
        with self._lock:
            self._current_gesture = gesture_name.lower().strip()
            self._last_update_time = time.time()

    def _draw_roboeyes_with_blink(self, draw):
        """Рисует глаза с простой логикой моргания внутри потока"""
        current_time = time.time()
        # Моргание каждые ~2.5 секунды на 0.15 секунды
        is_blinking = (current_time % 2.5) < 0.15 
        
        draw.rectangle((0, 0, 127, 63), fill="black") # Очистка
        
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

    def _display_loop(self):
        """Фоновый цикл отрисовки (~30 FPS), не блокирует основной поток"""
        frame_time = 1.0 / 30.0 
        
        while self._running:
            loop_start = time.time()
            
            with self._lock:
                gesture = self._current_gesture
                time_since_update = time.time() - self._last_update_time

            # Если жест не обновлялся дольше таймаута, переключаемся на RoboEyes
            if time_since_update > self._idle_timeout and gesture != "roboeyes":
                gesture = "roboeyes"

            if self.device:
                if gesture == "roboeyes":
                    # Программная отрисовка с морганием
                    from luma.core.render import canvas
                    with canvas(self.device) as draw:
                        self._draw_roboeyes_with_blink(draw)
                else:
                    # Мгновенная отрисовка из hex-массива
                    hex_data = get_pattern(gesture)
                    image = Image.frombytes('1', (128, 64), bytes.fromhex(hex_data))
                    self.device.display(image)

            # Контроль FPS, чтобы не грузить CPU на 100%
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    def cleanup(self):
        """Корректное завершение работы"""
        self._running = False
        if self._display_thread.is_alive():
            self._display_thread.join(timeout=1.0)
        if self.device:
            self.device.clear()
        print("[OLED] Дисплей очищен и поток остановлен.")