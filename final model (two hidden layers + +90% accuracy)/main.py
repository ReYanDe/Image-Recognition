"""Главный запуск (тестирования)"""
from utils.save_and_load_weight_bias import *
from model import feedforward
from utils.data_loader import random_img
import time


# загружаем веса и смещения 
weights = read_weight_hidden_def()
weights_hidden_2 = read_second_hidden_weight_def()
weights_output = read_output_weight_def()
bias_hidden = read_bias_hidden_def()
bias_hidden_2 = read_bias_second_def()
bias_output = read_bias_output_def()


for i in range(30):
    time.sleep(1)
    # нейроны изображения и правильное число
    img_neurons, true_number = random_img()


    # прямое аспространение
    a_hidden, a_hidden_2, z_output, res_softmax, predicted_number = feedforward(weights, weights_output, bias_hidden, bias_output, img_neurons, weights_hidden_2, bias_hidden_2)


    print(f"Предсказано: {predicted_number} | Правильно: {true_number}")    