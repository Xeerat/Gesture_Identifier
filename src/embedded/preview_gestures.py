"""Локальный предпросмотр SSD1306-паттернов без подключения OLED/I2C."""
import tkinter as tk

from patterns import GESTURE_PATTERNS
from PIL import Image, ImageDraw, ImageTk

SCREEN_SIZE = (128, 64)
SCALE = 5
GESTURES = ("stop", "rock", "palm", "one", "like", "four", "dislike", "roboeyes")


def make_frame(name: str) -> Image.Image:
    # Используем режим "L" (grayscale) вместо "1". 
    # Это гарантирует корректное отображение в ImageTk и стабильную отрисовку.
    # 0 = черный, 255 = белый
    screen = Image.new("L", SCREEN_SIZE, 0)
    
    if name == "roboeyes":
        draw = ImageDraw.Draw(screen)
        # Используем 255 для белого цвета
        draw.rounded_rectangle((16, 15, 53, 49), radius=8, outline=255, width=3)
        draw.rounded_rectangle((27, 24, 41, 39), radius=4, fill=255)
        draw.rounded_rectangle((75, 15, 112, 49), radius=8, outline=255, width=3)
        draw.rounded_rectangle((86, 24, 100, 39), radius=4, fill=255)
    else:
        # Загружаем 1-битный паттерн из hex
        icon = Image.frombytes("1", (64, 64), bytes.fromhex(GESTURE_PATTERNS[name]))
        # Конвертируем в "L". 
        # Трюк с .point() нужен, потому что при прямой конвертации "1" -> "L" 
        # значение 1 становится 1 (очень темно-серый, почти черный). 
        # Мы принудительно делаем все ненулевые пиксели ярко-белыми (255).
        icon = icon.convert("L").point(lambda p: 255 if p > 0 else 0)
        screen.paste(icon, (32, 0))
        
    # Масштабируем. Для режима "L" NEAREST работает идеально, сохраняя четкие пиксели
    return screen.resize((SCREEN_SIZE[0] * SCALE, SCREEN_SIZE[1] * SCALE), Image.Resampling.NEAREST)


class GesturePreview:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.index = 0
        self.image_label = tk.Label(root, bg="black", bd=0)
        self.image_label.pack(padx=20, pady=(20, 8))
        self.name_label = tk.Label(root, font=("TkDefaultFont", 14))
        self.name_label.pack(pady=(0, 12))
        controls = tk.Label(root, text="← / →: переключить   Пробел: пауза/продолжить   Esc: выход")
        controls.pack(padx=12, pady=(0, 16))
        root.bind("<Left>", lambda _event: self.show(self.index - 1))
        root.bind("<Right>", lambda _event: self.show(self.index + 1))
        root.bind("<space>", self.toggle)
        root.bind("<Escape>", lambda _event: root.destroy())
        self.paused = False
        self.photo = None
        self.show(0)
        self.root.after(5000, self.tick)

    def show(self, index: int):
        self.index = index % len(GESTURES)
        name = GESTURES[self.index]
        self.photo = ImageTk.PhotoImage(make_frame(name))
        self.image_label.configure(image=self.photo)
        self.name_label.configure(text=f"{name} — экран 128×64")

    def toggle(self, _event=None):
        self.paused = not self.paused
        if not self.paused:
            self.tick()

    def tick(self):
        if not self.paused:
            self.show(self.index + 1)
            self.root.after(5000, self.tick)


def main():
    root = tk.Tk()
    root.title("SSD1306 — предпросмотр жестов")
    root.resizable(False, False)
    GesturePreview(root)
    root.mainloop()


if __name__ == "__main__":
    main()
