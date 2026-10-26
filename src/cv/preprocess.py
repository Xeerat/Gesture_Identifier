"""
Предобработка для YOLO classification.
Геометрия повторяет classify_transforms из Ultralytics:
    Resize(bilinear) короткой стороны -> CenterCrop(imgsz).

Две функции выхода:
    preprocess_frame     -> float32 NCHW [0,1]      (для ONNX Runtime)
    preprocess_frame_rknn-> float32 NCHW [0,255]    (для RKNN, /255 встроено в модель)
"""

import cv2
import numpy as np


def _resize_short_side(img: np.ndarray, target: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h < w:
        new_h = target
        new_w = int(round(w * target / h))
    else:
        new_w = target
        new_h = int(round(h * target / w))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]

    if h < size or w < size:
        pad_top = max(0, (size - h) // 2)
        pad_bottom = max(0, size - h - pad_top)
        pad_left = max(0, (size - w) // 2)
        pad_right = max(0, size - w - pad_left)
        img = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        h, w = img.shape[:2]

    top = (h - size) // 2
    left = (w - size) // 2
    return img[top:top + size, left:left + size]


def _geometry(frame_bgr: np.ndarray, imgsz: int) -> np.ndarray:
    """BGR uint8 -> RGB uint8, resize + center crop. Возвращает HWC uint8."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = _resize_short_side(rgb, imgsz)
    return _center_crop(resized, imgsz)


def preprocess_frame(frame_bgr: np.ndarray, imgsz: int = 224) -> np.ndarray:
    """Для ONNX: float32 NCHW в диапазоне [0, 1]."""
    hwc = _geometry(frame_bgr, imgsz)
    chw = hwc.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw[None, ...], dtype=np.float32)


def preprocess_frame_rknn(frame_bgr: np.ndarray, imgsz: int = 224) -> np.ndarray:
    """Для RKNN: float32 NCHW в диапазоне [0, 255] (нормализация встроена в .rknn)."""
    hwc = _geometry(frame_bgr, imgsz)
    chw = hwc.transpose(2, 0, 1).astype(np.float32)
    return np.ascontiguousarray(chw[None, ...], dtype=np.float32)