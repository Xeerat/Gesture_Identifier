from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Загружает настройки из YAML-файла.

    Читает файл, превращает его содержимое в словарь Python
    и проверяет, что файл действительно содержит настройки (словарь),
    а не что-то другое.

    Args:
        path: Путь к файлу с настройками (можно строку или Path).

    Returns:
        dict[str, Any]: Словарь с настройками из файла.

    Raises:
        FileNotFoundError: Если файла нет по указанному пути.
        TypeError: Если файл есть, но внутри не словарь (например, пустой).
        yaml.YAMLError: Если в файле синтаксическая ошибка YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise TypeError(f"Конфиг {path} не является словарём")

    return config
