import random 
import cv2
import numpy as np 


def random_img():
    while True:
        # Рандомные числа для рандомной выборки чисел
        firstN = random.randint(0, 9)
        secondN = random.randint(1, 5000)

        # путь для выборки рандомных изображений
        path = f'MNISTDB/TrainingImages/{firstN}/{secondN}.png'
        # path_test = f'MNISTDB/TrainingImages/0/1.png'


        # чтение изображение
        a = cv2.imread(path)

        if a is None:
            continue

        # перевод изображения в серый цвет для определения оттенков серого
        gray_image = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

        # будет использоваться в том случае если изображение не нужного разрешения
        # small_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)

        # матрица цифр в виде отттенков серого от 0 - 1
        normalized = np.round(1.0 - (gray_image / 255.0), 4)

        # возвращаем изображение в виде сатрицы цифр, и правильный ответ
        return normalized.tolist(), firstN