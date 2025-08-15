"""Обучение нейросети с помощью нескольких десяток эпох"""
from model import feedforward, backpropagation
from utils.save_and_load_weight_bias import *

from utils.data_loader import random_img


epochs = 50000
learning_step = 0.01



# загружаем веса и смещения 
weights = read_weight_hidden_def()
weights_hidden_2 = read_second_hidden_weight_def()
weights_output = read_output_weight_def()
bias_hidden = read_bias_hidden_def()
bias_hidden_2 = read_bias_second_def()
bias_output = read_bias_output_def()

# переменные для просчета ошибки нейросети
total_loss = 0
correct_predictions = 0

for epoch in range(epochs):
    # нейроны изображения и правильное число
    img_neurons, true_number = random_img()

    # Прямое распространение
    a_hidden, a_hidden_2, z_output, res_softmax, predicted_number = feedforward(weights, weights_output, bias_hidden, bias_output, img_neurons, weights_hidden_2, bias_hidden_2)

    # обратное распространение
    weights, weights_hidden_2, weights_output, bias_hidden, bias_hidden_2, bias_output, loss = backpropagation(weights, bias_hidden, weights_hidden_2, bias_hidden_2, weights_output, bias_output,true_number, res_softmax, a_hidden, a_hidden_2, img_neurons,learning_step)

    total_loss += loss
    correct_predictions += int(predicted_number == true_number)

    # Каждые 100 эпох печатаем прогресс
    if (epoch + 1) % 100 == 0:
        avg_loss = total_loss / (epoch + 1)
        accuracy = correct_predictions / (epoch + 1) * 100
        print(f"Эпоха {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")                                                                             


save_weights_hidden_def(weights)
save_weights_hidden_2_def(weights_hidden_2)
save_weights_output_def(weights_output)
save_bias_hidden_def(bias_hidden)
save_bias_hidden_2_def(bias_hidden_2)
save_bias_output_def(bias_output)