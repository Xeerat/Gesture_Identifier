import time

from oled_controller import OLEDController


def main():
    # Укажи правильный порт I2C для твоей ROC-RK3588-PC (обычно 3 или 7)
    oled = OLEDController(i2c_port=3, i2c_address=0x3C)
    
    if not oled.device:
        print("Не удалось запустить тест. Проверьте подключение и права на /dev/i2c-*")
        return

    gestures = ["stop", "rock", "palm", "one", "like", "four", "dislike"]
    
    print("Начинаю тестирование hex-паттернов...")
    
    try:
        for gesture in gestures:
            print(f"Отображаю: {gesture}")
            oled.show_gesture(gesture)
            time.sleep(2) # Демонстрация жеста
            
        print("Переход в режим паузы (RoboEyes) на 5 секунд...")
        oled.show_gesture("roboeyes")
        time.sleep(5)
        
        print("Тест завершен успешно.")
        
    except KeyboardInterrupt:
        print("\nТест прерван.")
    finally:
        oled.cleanup()

if __name__ == "__main__":
    main()