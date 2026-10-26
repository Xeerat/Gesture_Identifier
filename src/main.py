from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from embedded.stub_led import StubLEDController
from interfaces import DetectorInterface, LEDControllerInterface

# Заглушки (позже заменятся реальными реализациями из cv/, ml/, embedded/)
from ml.stub_detector import StubDetector
from utils.config_loader import load_config
from utils.smoothing import GestureSmoother


# ============================================================
# Логирование
# ============================================================
def setup_logging(level_name: str) -> None:
    """
    Инициализирует и настраивает глобальную систему логирования приложения.

    Безопасно преобразует строковое название уровня логирования (например,
    "INFO" или "debug") в числовую константу модуля `logging`.
    Если передано некорректное или неизвестное значение, автоматически
    использует уровень INFO в качестве безопасного значения по умолчанию (fallback),
    предотвращая падение приложения.

    Args:
        level_name (str): Желаемый уровень логирования (например, "DEBUG",
                          "INFO", "WARNING", "ERROR"). Регистр символов не имеет значения.
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


logger = logging.getLogger("GestureControl")


# ============================================================
# Настройка и создание детектора и led-контроллера
# ============================================================
def create_detector(config: dict) -> DetectorInterface:
    """
    Функция для создания детектора жестов на основе конфигурации.

    Создаёт и возвращает экземпляр детектора в зависимости от параметра
    `model.backend` в конфиге. Поддерживает заглушку для тестирования
    и реальные бэкенды (RKNN, ONNX).

    Args:
        config (dict): Словарь с настройками приложения.

    Returns:
        DetectorInterface: Экземпляр детектора, реализующий интерфейс.

    Raises:
        ValueError: Если указан неподдерживаемый backend в конфиге.
    """

    backend = config["model"].get("backend", "stub")
    classes = config["model"].get("classes", ["none"])
    threshold = config["model"].get("confidence_threshold", 0.7)

    if backend == "stub":
        logger.info("Используется заглушка детектора (StubDetector)")
        return StubDetector(classes=classes, threshold=threshold)

    # TODO: подключить реальные реализации
    if backend == "rknn":
        from ml.rknn_detector import RKNN_detector
        return RKNN_detector(config["model"])
    if backend == "onnx":
        from ml.onnx_detector import ONNXDetector
        return ONNXDetector(config["model"])

    raise ValueError(f"Неизвестный backend модели: {backend}")


def create_led_controller(config: dict) -> LEDControllerInterface | None:
    """
    Функция для создания контроллера светодиодной ленты.

    Создаёт и возвращает экземпляр контроллера LED-ленты, если она
    включена в конфиге (`led.enabled: true`). Если лента отключена,
    возвращает None.

    Args:
        config (dict): Словарь с настройками приложения.

    Returns:
        LEDControllerInterface | None: Экземпляр контроллера ленты
            или None, если лента отключена в конфиге.

    Raises:
        ValueError: Если указан неподдерживаемый backend в конфиге.
    """

    if not config["led"].get("enabled", True):
        logger.info("LED-лента отключена в конфиге")
        return None

    backend = config["led"].get("backend", "stub")

    if backend == "stub":
        logger.info("Используется заглушка LED-контроллера (StubLEDController)")
        return StubLEDController()

    # TODO: подключить реальную реализацию
    # from src.embedded.firefly_led import FireflyLEDController
    # if backend == "fire":
    #     return FireflyLEDController(config["led"])

    raise ValueError(f"Неизвестный backend модели: {backend}")


# ============================================================
# Главный класс приложения
# ============================================================
class GestureControlApp:
    """
    Главный класс приложения «Жест-Контроль».

    Управляет жизненным циклом приложения, связывая воедино:
    захват видео с камеры, инференс модели распознавания,
    сглаживание результатов и управление светодиодной лентой.
    """

    def __init__(self, config: dict) -> None:
        """
        Инициализирует приложение и создаёт экземпляры всех компонентов.

        Args:
            config (dict): Словарь с настройками приложения, загруженный из YAML.
        """
        self.config = config
        self.running = False

        self._frame_times = []
        self._fps_window = 30

        logger.info("Инициализация компонентов...")
        self.detector = create_detector(config)
        self.led = create_led_controller(config)

        self.smoother = GestureSmoother(
            buffer_size=config["smoothing"]["buffer_size"],
            min_votes=config["smoothing"]["min_votes"],
            none_threshold=config["smoothing"].get("none_threshold", 5),
        )

        if self.led is not None:
            self.led.set_brightness(config["led"].get("brightness", 128))

        self.cap = None

    def _init_camera(self) -> None:
        """
        Инициализирует и настраивает захват видео с камеры.

        Устанавливает запрошенные разрешение и FPS, а также выполняет
        прогрев камеры, сбрасывая первые кадры для стабилизации.

        Raises:
            RuntimeError: Если не удалось открыть камеру по указанному device_id.
        """
        cam_cfg = self.config["camera"]
        device_id = cam_cfg.get("device_id", 0)
        width, height = cam_cfg.get("resolution", [640, 480])
        fps = cam_cfg.get("fps", 30)

        logger.info(f"Открытие камеры {device_id} ({width}x{height})")
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть камеру {device_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        warmup = cam_cfg.get("warmup_frames", 30)
        logger.info(f"Прогрев камеры ({warmup} кадров)...")
        for _ in range(warmup):
            self.cap.read()
        logger.info("Камера готова")

    def _update_fps(self, frame_time: float) -> float:
        """
        Вычисляет скользящее среднее значение FPS.

        Args:
            frame_time (float): Время обработки текущего кадра в секундах.

        Returns:
            float: Среднее количество кадров в секунду (FPS) за последний
                   интервал, заданный _fps_window.
        """
        self._frame_times.append(frame_time)
        if len(self._frame_times) > self._fps_window:
            self._frame_times.pop(0)
        avg_time = sum(self._frame_times) / len(self._frame_times)

        return 1.0 / avg_time if avg_time > 0 else 0.0

    def _draw_debug_info(
        self, frame: np.ndarray, gesture: str, confidence: float, fps: float
    ) -> None:
        """
        Рисует отладочную информацию поверх кадра изображения.

        Добавляет полупрозрачную подложку и текст с текущим жестом,
        уверенностью модели и FPS. Изменяет переданный массив numpy inplace.

        Args:
            frame (np.ndarray): Кадр изображения в формате BGR, который будет изменён.
            gesture (str): Текущий стабилизированный распознанный жест.
            confidence (float): Уверенность модели в распознавании (от 0.0 до 1.0).
            fps (float): Текущее значение FPS для отображения на экране.
        """
        _, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        text = f"Gesture: {gesture}  |  conf: {confidence:.2f}  |  FPS: {fps:.1f}"
        cv2.putText(
            frame, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

    def run(self) -> None:
        """
        Запускает основной бесконечный цикл обработки приложения.

        Выполняет последовательность действий для каждого кадра:
        чтение -> детекция -> фильтрация по порогу -> сглаживание ->
        обновление состояния LED -> отрисовка отладки -> ограничение FPS.

        Перехватывает KeyboardInterrupt (Ctrl+C) для корректного завершения.
        """
        self._init_camera()
        self.detector.warmup()

        self.running = True
        show_window = self.config["app"].get("show_debug_window", True)
        conf_threshold = self.config["model"].get("confidence_threshold", 0.7)
        target_fps = self.config["app"].get("target_fps", 15)
        frame_interval = 1.0 / target_fps if target_fps > 0 else 0

        last_led_gesture = "none"

        logger.info("=== Запуск главного цикла ===")
        if self.led is not None:
            self.led.show_idle()

        try:
            while self.running:
                loop_start = time.perf_counter()

                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Не удалось получить кадр, повтор...")
                    time.sleep(0.01)
                    continue

                result = self.detector.detect(frame)

                if result.confidence < conf_threshold:
                    raw_gesture = "none"
                else:
                    raw_gesture = result.gesture

                stable_gesture = self.smoother.process(raw_gesture)

                if self.led is not None and stable_gesture != last_led_gesture:
                    if stable_gesture == "none":
                        self.led.show_idle()
                    else:
                        self.led.show_gesture(stable_gesture)
                    last_led_gesture = stable_gesture

                if show_window:
                    frame_time = time.perf_counter() - loop_start
                    fps = self._update_fps(frame_time)
                    self._draw_debug_info(frame, stable_gesture, result.confidence, fps)
                    cv2.imshow("Gesture Control [DEBUG]", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):
                        logger.info("Получен сигнал выхода (ESC/q)")
                        self.running = False

                    if (
                        cv2.getWindowProperty(
                            "Gesture Control [DEBUG]", cv2.WND_PROP_VISIBLE
                        )
                        < 1
                    ):
                        logger.info("Окно закрыто пользователем (крестик)")
                        self.running = False

                elapsed = time.perf_counter() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Получен Ctrl+C, завершение...")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """
        Корректно освобождает все занятые системные ресурсы.

        Останавливает основной цикл, освобождает захват камеры,
        безопасно отключает светодиодную ленту и закрывает все окна OpenCV.
        """
        logger.info("Освобождение ресурсов...")
        self.running = False

        if self.cap is not None:
            self.cap.release()
            logger.info("Камера освобождена")

        if self.led is not None:
            self.led.close()

        cv2.destroyAllWindows()
        logger.info("=== Приложение остановлено ===")


# ============================================================
# Запуск
# ============================================================
def main() -> int:
    """
    Точка входа в приложение «Жест-Контроль».

    Выполняет инициализацию и запуск системы в следующем порядке:
    1. Загружает и валидирует конфигурацию из YAML-файла.
    2. Настраивает систему логирования на основе параметров конфига.
    3. Инициализирует главный класс приложения (камера, модель, LED).
    4. Запускает бесконечный цикл обработки видеопотока.

    Returns:
        int: Код завершения (exit code) для операционной системы.
             0 — приложение успешно завершено пользователем или штатно.
             1 — критическая ошибка на этапе инициализации
                 (например, файл конфигурации не найден или повреждён).
    """
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"❌ Файл конфигурации не найден: {e}", file=sys.stderr)
        return 1
    except (ValueError, yaml.YAMLError) as e:
        print(f"❌ Ошибка в формате конфигурации: {e}", file=sys.stderr)
        return 1

    setup_logging(config["app"].get("log_level", "INFO"))
    logger.info(f"Загружен конфиг: {config_path}")

    try:
        app = GestureControlApp(config)
        app.run()
    except ValueError as e:
        logger.error(f"Возникла ошибка при инициализации объекта: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Возникла ошибка при инициализации камеры: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())