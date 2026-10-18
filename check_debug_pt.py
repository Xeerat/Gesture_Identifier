#!/usr/bin/env python3
"""
Отдельный прогон сохранённых debug-кадров через best.pt.

Использует только raw-кадры вида frame_*.jpg из debug_pt/.
Цель: проверить, что модель реально отвечает на эти изображения.

Запуск из корня проекта:
    python check_debug_pt.py \
        --model src/ml/models/gesture_yolov8n.pt \
        --source-dir debug_pt \
        --pattern "frame_*.jpg" \
        --conf 0.001 \
        --iou 0.99 \
        --save-mode all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("DEBUG_FRAMES")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Прогон debug-кадров из debug_pt через best.pt"
    )

    parser.add_argument(
        "--model",
        default="src/ml/models/gesture_yolov8n.pt",
        help="Путь к best.pt",
    )

    parser.add_argument(
        "--source-dir",
        default="debug_pt",
        help="Папка с debug-кадрами",
    )

    parser.add_argument(
        "--pattern",
        default="frame_*.jpg",
        help="Шаблон файлов. ВАЖНО: использовать только raw кадры frame_*.jpg",
    )

    parser.add_argument(
        "--out-dir",
        default="debug_pt_check",
        help="Папка для сохранённых результатов проверки",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Порог уверенности для диагностики",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.99,
        help="Порог IoU для NMS. Для диагностики ставим почти 1.0",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Максимум кадров для проверки. 0 = все найденные",
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=1000,
        help="Максимум детекций на кадр",
    )

    parser.add_argument(
        "--save-mode",
        choices=["all", "detections", "none"],
        default="detections",
        help="Что сохранять: all = все кадры, detections = только с детекциями, none = не сохранять",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Только проверить, что изображения читаются нормально, без запуска модели",
    )

    return parser.parse_args()


def normalize_names(names: Any) -> Dict[int, str]:
    """
    Приводит model.names к виду {int: str}.
    """
    if names is None:
        return {}

    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    if isinstance(names, (list, tuple)):
        return {i: str(name) for i, name in enumerate(names)}

    return {}


def image_stats(img: np.ndarray) -> Dict[str, Any]:
    """
    Базовая статистика изображения, чтобы убедиться, что кадр не серый/не битый.
    """
    if img.ndim == 2:
        gray = img
        channels = 1
    elif img.ndim == 3:
        channels = img.shape[2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if channels >= 3 else img[:, :, 0]
    else:
        return {
            "shape": img.shape,
            "dtype": str(img.dtype),
            "channels": -1,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "edge_ratio": None,
            "warning": "unexpected_ndim",
        }

    mean = float(gray.mean())
    std = float(gray.std())
    min_v = int(gray.min())
    max_v = int(gray.max())

    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.count_nonzero(edges)) / float(edges.size) if edges.size > 0 else 0.0

    warning = None
    if channels not in (1, 3, 4):
        warning = f"unexpected_channels={channels}"
    elif img.dtype != np.uint8:
        warning = f"non_uint8_dtype={img.dtype}"
    elif max_v <= 1:
        warning = "image_maybe_normalized_float"
    elif std < 3.0:
        warning = "image_very_low_variance_possible_gray_or_blank"

    return {
        "shape": img.shape,
        "dtype": str(img.dtype),
        "channels": channels,
        "mean": round(mean, 3),
        "std": round(std, 3),
        "min": min_v,
        "max": max_v,
        "edge_ratio": round(edge_ratio, 5),
        "warning": warning,
    }


def tensor_to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def draw_detections(
    img: np.ndarray,
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    cls_ids: np.ndarray,
    names: Dict[int, str],
    title: str = "",
) -> np.ndarray:
    annotated = img.copy()

    for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confs, cls_ids):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cls_id = int(cls_id)
        conf = float(conf)

        label = names.get(cls_id, f"class_{cls_id}")

        # Зелёный — сильнее, жёлтый — слабый кандидат
        if conf >= 0.25:
            color = (0, 255, 0)
        elif conf >= 0.05:
            color = (0, 255, 255)
        else:
            color = (255, 255, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 2

        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - th - baseline - 6)),
            (x1 + tw, y1),
            color,
            -1,
        )

        cv2.putText(
            annotated,
            text,
            (x1, max(th, y1 - baseline - 3)),
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    if title:
        h, w = annotated.shape[:2]
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 44), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0, annotated)

        cv2.putText(
            annotated,
            title,
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def draw_no_detection(img: np.ndarray, title: str = "NO DETECTION") -> np.ndarray:
    annotated = img.copy()

    cv2.putText(
        annotated,
        title,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def collect_images(source_dir: Path, pattern: str, max_images: int) -> List[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Папка не найдена: {source_dir}")

    paths = sorted(source_dir.glob(pattern))

    if not paths:
        raise FileNotFoundError(
            f"В папке {source_dir} не найдены файлы по шаблону {pattern}. "
            f"Проверь, что используешь raw-кадры frame_*.jpg, а не annotated_*/no_det_*."
        )

    if max_images > 0:
        paths = paths[:max_images]

    return paths


def main() -> int:
    args = parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    model_path = Path(args.model)

    log.info("Source dir: %s", source_dir.resolve())
    log.info("Pattern: %s", args.pattern)
    log.info("Out dir: %s", out_dir.resolve())
    log.info("Model: %s", model_path.resolve())
    log.info("conf=%.6f iou=%.6f max_det=%d save_mode=%s check_only=%s",
             args.conf, args.iou, args.max_det, args.save_mode, args.check_only)

    if args.save_mode != "none":
        out_dir.mkdir(parents=True, exist_ok=True)

    paths = collect_images(source_dir, args.pattern, args.max_images)
    log.info("Найдено кадров: %d", len(paths))

    # Предупреждение, если человек случайно взял annotated/no_det
    bad_names = [p.name for p in paths if p.name.startswith(("annotated_", "no_det_"))]
    if bad_names:
        log.warning(
            "ВНИМАНИЕ: в выборке есть файлы с overlay: %s. "
            "Для честной проверки используй только frame_*.jpg",
            bad_names[:10],
        )

    model = None
    names: Dict[int, str] = {}

    if not args.check_only:
        if not model_path.exists():
            log.error("Модель не найдена: %s", model_path)
            return 2

        try:
            from ultralytics import YOLO
        except ImportError:
            log.exception("Не установлен ultralytics. Поставь CPU-версию torch + ultralytics.")
            return 2

        log.info("Загружаю модель: %s", model_path)
        model = YOLO(str(model_path))

        raw_names = getattr(model, "names", None)
        if raw_names is None:
            raw_names = getattr(model.model, "names", {})

        names = normalize_names(raw_names)
        log.info("Классы модели: %s", names)

    images_checked = 0
    images_with_det = 0
    total_dets = 0
    global_class_dist: Dict[int, int] = {}

    for idx, path in enumerate(paths, start=1):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if img is None:
            log.error("[%03d] НЕ УДАЛОСЬ ПРОЧИТАТЬ: %s", idx, path)
            continue

        images_checked += 1
        stats = image_stats(img)

        log.info("[%03d] %s stats=%s", idx, path.name, stats)

        if stats.get("warning"):
            log.warning("[%03d] ПРОБЛЕМНОЕ ИЗОБРАЖЕНИЕ: %s", idx, stats["warning"])

        if args.check_only:
            continue

        assert model is not None

        results = model.predict(
            source=img,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            verbose=False,
            device="cpu",
        )

        if not results:
            log.warning("[%03d] results пустой", idx)
            if args.save_mode == "all":
                out_path = out_dir / f"empty_{path.stem}.jpg"
                cv2.imwrite(str(out_path), draw_no_detection(img, "EMPTY RESULTS"))
            continue

        result = results[0]
        boxes = getattr(result, "boxes", None)

        if boxes is None or len(boxes) == 0:
            log.info("[%03d] NO DETECTION", idx)
            if args.save_mode == "all":
                out_path = out_dir / f"nodet_{path.stem}.jpg"
                cv2.imwrite(str(out_path), draw_no_detection(img))
            continue

        confs = tensor_to_numpy(boxes.conf).astype(np.float32)
        cls_ids = tensor_to_numpy(boxes.cls).astype(np.int64)
        boxes_xyxy = tensor_to_numpy(boxes.xyxy).astype(np.float32)

        images_with_det += 1
        total_dets += len(confs)

        local_dist: Dict[int, int] = {}
        for cls_id in cls_ids.tolist():
            cls_id = int(cls_id)
            local_dist[cls_id] = local_dist.get(cls_id, 0) + 1
            global_class_dist[cls_id] = global_class_dist.get(cls_id, 0) + 1

        # Сортируем по убыванию conf
        order = np.argsort(confs)[::-1]
        top_n = min(10, len(order))
        top_list: List[Tuple[int, str, float, Tuple[int, int, int, int]]] = []

        for oi in order[:top_n]:
            cls_id = int(cls_ids[oi])
            conf = float(confs[oi])
            x1, y1, x2, y2 = map(int, boxes_xyxy[oi])
            label = names.get(cls_id, f"class_{cls_id}")
            top_list.append((cls_id, label, round(conf, 6), (x1, y1, x2, y2)))

        log.info(
            "[%03d] DETECTIONS count=%d local_dist=%s top=%s",
            idx,
            len(confs),
            local_dist,
            top_list,
        )

        if args.save_mode in ("all", "detections"):
            title = f"{path.name} dets={len(confs)} dist={local_dist}"
            annotated = draw_detections(
                img=img,
                boxes_xyxy=boxes_xyxy,
                confs=confs,
                cls_ids=cls_ids,
                names=names,
                title=title,
            )
            out_path = out_dir / f"det_{path.stem}.jpg"
            cv2.imwrite(str(out_path), annotated)

    log.info("========== SUMMARY ==========")
    log.info("Изображений проверено: %d", images_checked)
    log.info("Изображений с детекциями: %d", images_with_det)
    log.info("Всего детекций: %d", total_dets)
    log.info("Глобальное распределение class_id: %s", global_class_dist)

    if names and global_class_dist:
        named_dist = {
            f"{cid}:{names.get(cid, '?')}": count
            for cid, count in sorted(global_class_dist.items())
        }
        log.info("Глобальное распределение по именам: %s", named_dist)

    if args.save_mode != "none":
        log.info("Результаты сохранены в: %s", out_dir.resolve())

    return 0


if __name__ == "__main__":
    sys.exit(main())