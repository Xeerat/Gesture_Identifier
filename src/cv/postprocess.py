"""
Модуль постобработки выходных данных YOLOv8.
Универсальная версия: работает и с 7-классовой ([1,11,8400]),
и с 18-классовой ([1,22,8400]) моделью, и с обоими layout'ами
([features, anchors] или [anchors, features]).
Автоприменяет sigmoid, если scores выглядят как logits.
Добавлены отладочные логи на каждом этапе.
"""
from typing import List, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _prepare_yolov8_preds(output: np.ndarray) -> np.ndarray:
    """
    Приводит выход YOLOv8 к виду [anchors, features].
    Поддерживает:
      [1, 11, 8400] -> [8400, 11]
      [1, 8400, 11] -> [8400, 11]
      [11, 8400]    -> [8400, 11]
      [8400, 11]    -> [8400, 11]
    """
    arr = np.asarray(output)
    logger.info("[POSTPROCESS] raw output shape=%s dtype=%s", arr.shape, arr.dtype)

    if arr.ndim == 3:
        arr = arr[0]
        logger.info("[POSTPROCESS] после squeeze batch: shape=%s", arr.shape)

    if arr.ndim != 2:
        logger.warning("[POSTPROCESS] unexpected ndim=%s, shape=%s", arr.ndim, arr.shape)
        return np.array([])

    # Автоопределение layout: если признаков меньше чем anchor'ов — транспонируем
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
        logger.info("[POSTPROCESS] транспонировано -> shape=%s", arr.shape)
    else:
        logger.info("[POSTPROCESS] layout уже [anchors, features] -> shape=%s", arr.shape)

    return arr


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    logger.info("[POSTPROCESS] NMS: вошло=%d вышло=%d iou_thresh=%.3f", len(boxes), len(keep), iou_thresh)
    return keep


def decode_yolov8_output(
    output: np.ndarray,
    conf_thresh: float = 0.45,
    iou_thresh: float = 0.45,
) -> np.ndarray:
    """
    Универсальное декодирование выхода YOLOv8.
    Возвращает массив [N, 6]: x1, y1, x2, y2, confidence, class_id.
    """
    preds = _prepare_yolov8_preds(output)

    if preds.size == 0:
        logger.warning("[POSTPROCESS] preds пустой после prepare")
        return np.array([])

    if preds.ndim != 2 or preds.shape[1] < 5:
        logger.warning("[POSTPROCESS] некорректная форма preds: %s", preds.shape)
        return np.array([])

    n_classes = preds.shape[1] - 4
    logger.info("[POSTPROCESS] определено классов: %d (shape[1]=%d)", n_classes, preds.shape[1])

    boxes_xywh = preds[:, :4].astype(np.float32)
    class_scores_raw = preds[:, 4:].astype(np.float32)

    logger.info(
        "[POSTPROCESS] boxes_xywh shape=%s class_scores shape=%s",
        boxes_xywh.shape, class_scores_raw.shape,
    )
    logger.info(
        "[POSTPROCESS] class_scores_raw: min=%.4f max=%.4f mean=%.4f",
        float(class_scores_raw.min()),
        float(class_scores_raw.max()),
        float(class_scores_raw.mean()),
    )

    if class_scores_raw.size == 0:
        logger.warning("[POSTPROCESS] class_scores пустой")
        return np.array([])

    # Авто-активация: если значения вне [0,1] — это logits, гоним через sigmoid
    min_s = float(class_scores_raw.min())
    max_s = float(class_scores_raw.max())

    if min_s < 0.0 or max_s > 1.0:
        class_scores = _sigmoid(class_scores_raw)
        logger.info(
            "[POSTPROCESS] применён SIGMOID (logits: min=%.4f max=%.4f) -> после: min=%.4f max=%.4f",
            min_s, max_s,
            float(class_scores.min()), float(class_scores.max()),
        )
    else:
        class_scores = class_scores_raw
        logger.info("[POSTPROCESS] sigmoid НЕ применён (уже probabilities: min=%.4f max=%.4f)", min_s, max_s)

    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    logger.info(
        "[POSTPROCESS] max_scores по всем anchor'ам: min=%.4f max=%.4f mean=%.4f median=%.4f",
        float(max_scores.min()),
        float(max_scores.max()),
        float(max_scores.mean()),
        float(np.median(max_scores)),
    )

    # Сколько anchor'ов выше порога
    mask = max_scores > conf_thresh
    n_total = len(max_scores)
    n_pass = int(mask.sum())
    logger.info(
        "[POSTPROCESS] фильтр conf_thresh=%.3f: прошло %d из %d anchor'ов",
        conf_thresh, n_pass, n_total,
    )

    if n_pass == 0:
        logger.info("[POSTPROCESS] ни один anchor не прошёл порог -> no_detection")
        return np.array([])

    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    # xywh -> xyxy
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0

    logger.info("[POSTPROCESS] после xywh->xyxy: boxes shape=%s", boxes_xyxy.shape)

    keep = _nms(boxes_xyxy, max_scores, iou_thresh)

    if len(keep) == 0:
        logger.info("[POSTPROCESS] NMS убил всё -> no_detection")
        return np.array([])

    result = np.column_stack([
        boxes_xyxy[keep],
        max_scores[keep],
        class_ids[keep].astype(np.float32),
    ])

    logger.info("[POSTPROCESS] ИТОГО боксов после декодирования+NMS: %d", len(result))

    # Печатаем топ-3 для наглядности
    if len(result) > 0:
        top_idx = np.argsort(-result[:, 4])[:3]
        for rank, idx in enumerate(top_idx):
            logger.info(
                "[POSTPROCESS]   top%d: conf=%.4f class_id=%d bbox=(%.1f, %.1f, %.1f, %.1f)",
                rank + 1,
                float(result[idx, 4]),
                int(result[idx, 5]),
                float(result[idx, 0]),
                float(result[idx, 1]),
                float(result[idx, 2]),
                float(result[idx, 3]),
            )

    return result


def rescale_boxes(
    boxes: np.ndarray,
    ratio: Tuple[float, float],
    pad: Tuple[float, float],
    orig_shape: Tuple[int, int],
) -> np.ndarray:
    """Пересчёт координат из 640x640 в исходный кадр."""
    if len(boxes) == 0:
        return boxes

    dw, dh = pad
    rw, rh = ratio

    boxes = boxes.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / rw
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / rh

    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

    logger.info(
        "[POSTPROCESS] rescale_boxes: ratio=%s pad=%s orig_shape=%s -> boxes shape=%s",
        ratio, pad, orig_shape, boxes.shape,
    )

    return boxes