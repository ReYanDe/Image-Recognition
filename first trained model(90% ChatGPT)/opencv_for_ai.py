import cv2
import numpy as np



def img(a):

    b = cv2.imread(a)

    if b is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {a}")

    gray_image = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

    small_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)

    normalized = np.round(1.0 - (small_image / 255.0), 1)

    # print("Оттенки пикселей (0.0 = белый, 1.0 = черный):")
    # for row in normalized:
    #     print(" ".join(f"{val:.2f}" for val in row))

    # val = normalized[1]
    # print(f'Значение 1 столбца чисел {val}')

    # preview_image = cv2.resize(small_image, (256, 256), interpolation=cv2.INTER_NEAREST)
    # cv2.imshow("Grayscale 20x20 Image (Upscaled)", preview_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return normalized

# пример пользования
# image = cv2.imread("images/ph1.jpg")  # Замените на путь к вашему файлу
# a = img(image)
# print(a[1])
