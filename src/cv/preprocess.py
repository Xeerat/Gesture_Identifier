"""
Модуль предобработки кадров для модели распознавания жестов.
Логика предоставлена ML-инженером, математика не меняется.
Добавлены отладочные логи на каждом этапе.
"""
from typing import Tuple
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Resize с сохранением пропорций + padding."""
    shape = img.shape[:2]  # [h, w]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    logger.info(
        "[PREPROCESS] letterbox: orig_shape=%s ratio=%.4f new_unpad=%s pad=(dw=%.1f, dh=%.1f)",
        shape, r, new_unpad, dw, dh,
    )

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        logger.info("[PREPROCESS] после resize: shape=%s", img.shape)
    else:
        logger.info("[PREPROCESS] resize не потребовался")

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    logger.info(
        "[PREPROCESS] после padding: shape=%s borders=(top=%d,bottom=%d,left=%d,right=%d)",
        img.shape, top, bottom, left, right,
    )

    return img, (r, r), (dw, dh)


def preprocess_frame(
    frame_bgr: np.ndarray,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Полная предобработка кадра.
    Input:  BGR кадр с камеры.
    Output: (RGB uint8 [640,640,3], ratio, pad)
    """
    logger.info(
        "[PREPROCESS] ВХОД: shape=%s dtype=%s min=%d max=%d",
        frame_bgr.shape, frame_bgr.dtype,
        int(frame_bgr.min()), int(frame_bgr.max()),
    )

    img, ratio, pad = letterbox(frame_bgr, new_shape=(640, 640))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    logger.info(
        "[PREPROCESS] ВЫХОД: shape=%s dtype=%s min=%d max=%d ratio=%s pad=%s",
        img_rgb.shape, img_rgb.dtype,
        int(img_rgb.min()), int(img_rgb.max()),
        ratio, pad,
    )

    return img_rgb, ratio, pad