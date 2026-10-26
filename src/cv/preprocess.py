import cv2
import numpy as np


def _resize_short_side(img: np.ndarray, target: int) -> np.ndarray:
    """
    Масштабирует изображение, сохраняя пропорции, так чтобы меньшая сторона
    стала равна заданному размеру.

    Args:
        img (np.ndarray): Входное изображение (HWC).
        target (int): Целевой размер меньшей стороны в пикселях.

    Returns:
        np.ndarray: Масштабированное изображение. Большая сторона будет
                    пропорционально увеличена или уменьшена.
    """
    h, w = img.shape[:2]
    if h < w:
        new_h = target
        new_w = round(w * target / h)
    else:
        new_w = target
        new_h = round(h * target / w)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """
    Выполняет центральное кадрирование (crop) изображения до размера size x size.

    Если исходное изображение меньше заданного размера, оно предварительно
    дополняется черными границами (padding) до нужных размеров, чтобы избежать
    ошибок индексации.

    Args:
        img (np.ndarray): Входное изображение (HWC).
        size (int): Целевой размер ширины и высоты для кадрирования.

    Returns:
        np.ndarray: Кадрированное изображение размера (size, size, C).
    """
    h, w = img.shape[:2]

    if h < size or w < size:
        pad_top = max(0, (size - h) // 2)
        pad_bottom = max(0, size - h - pad_top)
        pad_left = max(0, (size - w) // 2)
        pad_right = max(0, size - w - pad_left)
        img = cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        h, w = img.shape[:2]

    top = (h - size) // 2
    left = (w - size) // 2
    return img[top : top + size, left : left + size]


def _geometry(frame_bgr: np.ndarray, imgsz: int) -> np.ndarray:
    """
    Внутренняя функция, выполняющая базовые геометрические преобразования кадра.

    Последовательность: конвертация BGR в RGB -> ресайз по короткой стороне
    -> центральное кадрирование.

    Args:
        frame_bgr (np.ndarray): Входной кадр в цветовом пространстве BGR (uint8).
        imgsz (int): Целевой размер выходного квадратного изображения.

    Returns:
        np.ndarray: Обработанное изображение в формате RGB, shape (H, W, C),
                    тип данных uint8.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = _resize_short_side(rgb, imgsz)
    return _center_crop(resized, imgsz)


def preprocess_frame(frame_bgr: np.ndarray, imgsz: int = 224) -> np.ndarray:
    """
    Полный пайплайн предобработки кадра для инференса моделей через **ONNX Runtime**.

    Выполняет геометрические преобразования, меняет порядок осей с HWC на NCHW,
    преобразует тип данных в float32 и нормализует значения пикселей к диапазону [0.0, 1.0].

    Args:
        frame_bgr (np.ndarray): Входной кадр с камеры (обычно BGR, uint8).
        imgsz (int, optional): Целевой размер изображения. По умолчанию 224.

    Returns:
        np.ndarray: Готовый тензор для ONNX формы (1, C, H, W), тип float32,
                    непрерывный в памяти (C-contiguous), значения в диапазоне [0.0, 1.0].
    """
    hwc = _geometry(frame_bgr, imgsz)
    chw = hwc.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw[None, ...], dtype=np.float32)


def preprocess_frame_rknn(frame_bgr: np.ndarray, imgsz: int = 224) -> np.ndarray:
    """
    Полный пайплайн предобработки кадра для инференса моделей через **RKNN (NPU Rockchip)**.

    Выполняет геометрические преобразования, меняет порядок осей с HWC на NCHW
    и преобразует тип данных в float32.

    ВАЖНО: В отличие от ONNX-версии, здесь НЕТ деления на 255.0. Нормализация
    (mean/std) должна быть встроена в саму .rknn модель на этапе конвертации
    (через параметры `mean_values` и `norm_values` в RKNN-Toolkit).

    Args:
        frame_bgr (np.ndarray): Входной кадр с камеры (обычно BGR, uint8).
        imgsz (int, optional): Целевой размер изображения. По умолчанию 224.

    Returns:
        np.ndarray: Готовый тензор для RKNN формы (1, C, H, W), тип float32,
                    непрерывный в памяти (C-contiguous), значения в диапазоне [0.0, 255.0].
    """
    hwc = _geometry(frame_bgr, imgsz)
    chw = hwc.transpose(2, 0, 1).astype(np.float32)
    return np.ascontiguousarray(chw[None, ...], dtype=np.float32)
