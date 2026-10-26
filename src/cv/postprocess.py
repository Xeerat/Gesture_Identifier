from collections.abc import Sequence

import numpy as np


def decode_classification_output(
    output: np.ndarray,
    classes: Sequence[str],
    confidence_threshold: float = 0.5,
) -> tuple[str, float, int]:
    """
    Декодирует сырой вывод классификационной модели в читаемый формат.

    Автоматически преобразует входной тензор в одномерный массив вероятностей,
    находит класс с максимальной уверенностью и применяет пороговую фильтрацию.
    Также защищает от выхода индекса класса за границы предоставленного списка.

    Args:
        output (np.ndarray): Сырой вывод модели (логиты или вероятности).
                             Функция автоматически применяет `.reshape(-1)` для
                             гарантии одномерности.
        classes (Sequence[str]): Список или кортеж имен классов, где индекс
                                 элемента соответствует индексу выхода модели.
        confidence_threshold (float, optional): Минимальный порог уверенности
                             (от 0.0 до 1.0) для принятия предсказания.
                             По умолчанию 0.5.

    Returns:
        tuple[str, float, int]: Кортеж, содержащий три элемента:
            - gesture (str): Название предсказанного класса. Возвращает `"none"`,
              если уверенность ниже порога, или `"class_{id}"`, если индекс класса
              выходит за пределы списка `classes`.
            - confidence (float): Уверенность модели (максимальная вероятность).
              Возвращает `0.0`, если сработал порог отсечения.
            - class_id (int): Индекс предсказанного класса в массиве вероятностей.
              Возвращает `-1`, если сработал порог отсечения.
    """
    # Модель уже отдает вероятности
    probs = np.asarray(output, dtype=np.float32).reshape(-1)

    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])

    if confidence < confidence_threshold:
        return "none", 0.0, -1

    if 0 <= class_id < len(classes):
        gesture = str(classes[class_id])
    else:
        gesture = f"class_{class_id}"

    return gesture, confidence, class_id
