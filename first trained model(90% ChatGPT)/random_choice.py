import random
from opencv_for_ai import img
import cv2
import numpy as np



def random_img():
    while True:
        firstN = random.randint(0, 9)

        secondN = random.randint(1, 5000)

        path = f"images/TrainingImages/{firstN}/{secondN}.png"

        a = cv2.imread(path)

        if a is None:
            continue

        # if b is None:
        #     raise FileNotFoundError(f"Не удалось загрузить изображение: {a}")

        gray_image = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

        small_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)

        normalized = np.round(1.0 - (small_image / 255.0), 1)


        # print(f"Попытка загрузить изображение: images/TrainingImages/{firstN}/{secondN}.png")

        return normalized.tolist(), firstN





# image_path = r"C:\Users\User\Desktop\ai\images\TrainingImages\{0}\{1}.png".format(firstN, secondN)

# os.startfile(image_path)