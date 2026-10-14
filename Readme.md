# 🖐️ Жест-Контроль

Система распознавания жестов в реальном времени на плате Firefly с выводом
визуальных паттернов на светодиодную матрицу.

## 🏗 Архитектура

**Camera → Preprocessing → [NPU model] → Postprocessing → Smoother → LED matrix**


## 🚀 Быстрый старт

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск
cd src/
python main.py
```