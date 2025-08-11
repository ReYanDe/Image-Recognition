"""Архиетктура нейросети (feedforward, backpropagation)"""
import numpy as np
import random 


from utils.save_and_load_weight_bias import *
from utils.data_loader import random_img
from utils.math_utils import *


# веса для 64 нейронов скрытого слоя
weights = read_weight_hidden_def()

# веса для 10 нейронов выходного слоя
output_weight = read_output_weight_def()

# смещения весов скрытого слоя
bias_hidden = read_bias_hidden_def()

# смещения весов выходного слоя
bias_output = read_bias_output_def()

# нейроны картинки и правильне загаданное число 
img_neurons, true_number = random_img()


def feedforward(weights, output_weight, bias_hidden, bias_output, img_neurons):

    """-----------------------------------------------FEEDFORWARD-------------------------------------------------------------"""
    # список в который будет хранится резервная копия отыетов от 64 нейронов
    a_hidden = []

    """Применяем функцию активации для скрытого слоя"""

    for i in range(64):
        # тут происходит математическое вычисление по формуле z = a[i] * W[i][j] + bias[i]
        z = scalar(img_neurons, weights[i]) + bias_hidden[i]
        # применяем функцию активации сигмоид
        # sigmoid = 1 / ( 1 + math.exp(-z))
        #  Ограничиваем диапазон, чтобы избежать переполнения
        if z < -700:
            z = -700
        elif z > 700:
            z = 700
        sigmoid = 1 / (1 + math.exp(-z))

        a_hidden.append(sigmoid)


    """применяем функцию активации для выходного слоя"""

    # z это отображение от формулы
    z_output = []

    for j in range(10):
        # создаем связь между скрытым слоем и выходным и добавляем смящения
        z = scalar_output_z(a_hidden, output_weight[j]) + bias_output[j]
        z_output.append(z)


    res_softmax = softmax(z_output)
    predicted_number = res_softmax.index(max(res_softmax))

    return a_hidden, z_output, res_softmax, predicted_number

# print(f'Нейросеть думает что правильное число это {predicted_number} правильное число {true_number}')

"""-----------------------------------------------BackPropagation-------------------------------------------------------------"""


def backpropagation(weights, bias_hidden, output_weight, bias_output,  true_number ,res_softmax, a_hidden, img_neurons, learning_step, predicted_number):

    # список в котором будет хранится значения того насколько нейросеть ошиблась в вычислении
    delta_output = []

    # шаг обучения это то с какой скоростью будет оучаться нейросеть 0.1 = 10%; 0.01 = 1% и тд 
    # learning_step = 0.05

    """Вычисляем ошибку на выходе. То есть если число является правильным ответом его уменьшаем чтобы не мешался а оставльные добавляем в список чтобы в дальнейшем взять эти числа и с помощью них найти насколько оно ошиблось и исправить ошибку"""
    for i in range(10):
        if i != true_number:
            delta_output.append(res_softmax[i])
        else:
            delta_output.append(res_softmax[i] - 1)

    """Обновляем веса и смещения выходнога слоя"""

    for j in range(10):
        for i in range(64):
            # вычисляем то насколько нейросеть ошиблась и насколько нужно сместить весы 
            change_weight = delta_output[j] * a_hidden[i]
            # подправляем весы выходного слоя с учетом шага обучения
            output_weight[j][i] -= learning_step * change_weight
        # редактируем смящения выходного слоя нейронов
        bias_output[j] -= learning_step * delta_output[j]


    """Cчитываем насколько нейросеть ошиблась в своих вычислениях для скрытого слоя"""

    # список в котором будет хранится значения ошибки нейрона скрытого слоя
    delta_hidden = []

    for i in range(64):
        # переменная в которой будет накапливаться сумма ошибок
        error_sum = 0
        for j in range(10):
            # сумируем все ошибки скрытого слоя
            # Чтобы понять, насколько скрытый нейрон виноват в общей ошибке, мы и суммируем 
            error_sum += delta_output[j] * output_weight[j][i]

        sigmoid_derivate = a_hidden[i] * (1 - a_hidden[i])
        delta_hidden.append(error_sum * sigmoid_derivate)
            
        delta_hidden[i] = max(min(delta_hidden[i], 10), -10)

        bias_hidden[i] -= learning_step * delta_hidden[i]


    """Изменияем веса нейронов скрытого слоя"""

    for i in range(64):
        for row in range(28):
            for col in range(28):
                weight_change = delta_hidden[i] * img_neurons[row][col]
                weights[i][row][col] -= weight_change * learning_step


    if res_softmax[true_number] <= 0:
        res_softmax[true_number] = 1e-9

    loss = -math.log(res_softmax[true_number])

    return weights, output_weight, bias_hidden, bias_output, loss
