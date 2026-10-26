"""
Постобработка для YOLO classification.
Модель уже отдает вероятности, softmax не нужен.
"""

from typing import Sequence, Tuple
import numpy as np


def decode_classification_output(
    output: np.ndarray,
    classes: Sequence[str],
    confidence_threshold: float = 0.5,
) -> Tuple[str, float, int]:
    """
    Декодирует выход classification-модели.
    
    Returns:
        (gesture, confidence, class_id)
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