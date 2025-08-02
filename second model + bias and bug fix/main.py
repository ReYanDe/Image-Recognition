from image_random import random_img 
from utils import scalar, softmax
import random 
import numpy as np
import math





# # Создается веса для 64 нейронов сразу в переменную weights
# weights = [[[round(random.uniform(-0.1, 0.1) , 4) for _ in range(28)] for _ in range(28)] for _ in range(64)]

# """Создание рандомных чисел для матрицы весов состоящая из 64 нейронов"""

# with open('db/weight.txt', 'w') as w:
#     # 64 раза проходимся по циклу для 64 нейронов
#     for i in range(64):
#         # проверяем каждый элемент в матрице
#         for row in weights[i]:
#             # переобразуем список в простые числа, добавляем пробелы для каждой матрицы
#             w.write(' '.join(map(str, row)) + '\n')

# """Создание рандомных чисел для матрицы весов состоящая из 64 готовых ответа и 10 нейронов выходного слоя"""

# # создание рандомных чисел для весов 10х64 
# weights_output = [[round(random.uniform(-0.1, 0.1) , 4) for _ in range(64)] for _ in range(10)]


# with open('db/weight_output.txt', 'w') as w:
#     # проверяем каждый список индивидуально
#     for row in weights_output:
#         # вписываем каждое число без запятых только по пробелу и в конце вставляем отступ
#         w.write(' '.join(map(str, row)) + '\n')


# """Создания смещений (bias) для скрытого слоя (64 нейрона)"""


# # создание рандомных чисел для смещений (64) 
# bias_hidden = [round(random.uniform(-0.1, 0.1) , 4) for _ in range(64)]


# with open('db/bias_hidden.txt', 'w') as w:
#     # проверяем каждый список индивидуально
#     for b in bias_hidden:
#         # вписываем каждое число без запятых только по пробелу и в конце вставляем отступ
#         w.write(str(b) + '\n')


# """Создания смещений (bias) для скрытого слоя (10 нейрона)"""


# # создание рандомных чисел для смещений (10) 
# bias_output = [round(random.uniform(-0.1, 0.1) , 4) for _ in range(10)]


# with open('db/bias_output.txt', 'w') as w:
#     # проверяем каждый список индивидуально
#     for b in bias_output:
#         # вписываем каждое число без запятых только по пробелу и в конце вставляем отступ
#         w.write(str(b) + '\n')





"""Открытие матрицы весов состоящая из 64 нейронов"""

weights = []
            
with open('db/weight.txt', 'r') as r:
    current_neuron = []
    # итерируем и получаем пара индекс, значение 
    for i, line in enumerate(r):
        # strip() - удаляет пробелы и табуляции, split() - превращает в список 
        row = list(map(float, line.strip().split()))
        current_neuron.append(row)
        # Проверяем есть ли в данной строке 28 элемент для одного списка
        if len(current_neuron) == 28:
            weights.append(current_neuron)
            # очищаем список
            current_neuron = []


"""Чтение веса выходного слоя 10 нейронов """

weights_output = []

with open('db/weight_output.txt', 'r') as r:
    # проходимся по списку элементов 10 нейронов
    for i, line in enumerate(r):
        # очищаем от лишних пробелов и превращаем в список
        row = list(map(float, line.strip().split()))
        weights_output.append(row)

"""Чтение смещений (bias) входного слоя 64 нейронов """

bias_hidden = []

with open('db/bias_hidden.txt', 'r') as r:
    # проходимся по списку элементов 64 нейронов
    for i, line in enumerate(r):
        # очищаем от лишних пробелов и превращаем в список
        bias_hidden.append(float(line.strip()))


"""Чтение смещений (bias) входного слоя 64 нейронов """

bias_output = []

with open('db/bias_output.txt', 'r') as r:
    # проходимся по списку элементов 10 нейронов
    for i, line in enumerate(r):
        # очищаем от лишних пробелов и превращаем в список
        bias_output.append(float(line.strip()))


# """
# ---------------------------------------------------------   FeedForward       ----------------------------------------------------------------
# Прямое распространение нейросети это когда одна волна полностья взяла и прошлась по скрытым слоям (при наличии нескольких скрытых слоев)
# и дошла до выходного слоя.
# """


total_loss = 0
correct = 0


for epoch in range(1100):


    # img_neurons - это матрица чисел обозначающие яркость пикселя; true_number - нарисованное число
    img_neurons, true_number = random_img()

    

    # создаю список из 64 одниковых значений
    # b_hiden = [0 for i in range(64)]

    # список куда будут сохранятся ответы от 64 нейронов скрытого слоя
    a_hidden = []
    # список в котором будет хранится значение скрытых нейронов резервно до функции активации для использования в дальнейшем в коде
    z_hidden = []

    for i in range(64):
        # находим z по формуле z = W[i] * a[i] + b[i]
        z = scalar(img_neurons, weights[i]) + bias_hidden[i]
        # резервно сохраняем значения z в списке для дальнейшего использования
        z_hidden.append(z)
        # активируем нейроны благодаря функции активации ReLU
        ReLU = max(0, z)
        a_hidden.append(ReLU)



    # b_output = [0 for i in range(10)]
    # z - это обозначения от формулы
    z_output = []

    # 10 раз обрабатываем 10 нейронов
    for j in range(10): 
        z = 0
        # обрабатываем 64 цифры находящиеся в списке вместе с 10 нейронами
        for i in range(64):
            # перемнаживаем между собой ответ от скрытого слоя с весами выходного слоя (с каждым из них) их на секундочку 64х10
            # тоесть каждый ответ от скрытого слоя перемнаживаем с каждым из 10 нейронов выходного слоя 
            z += a_hidden[i] * weights_output[j][i] 
        # добавляем смищения выходного слоя
        z += bias_output[j]
        z_output.append(z)

    # теперь чтобы знать какой результат в процентном соотношении нам нужно 
    # список с 10 числами обработать в функции softmax которая в свою очередь вернет значения в працентном соотношении
    res_softmax = softmax(z_output)
    predicted_number = res_softmax.index(max(res_softmax))



    """Проверка работоспособности прямого распространения"""


    # for i, prob in enumerate(res_softmax):
    #     print(f'Цифра:{i}; вероятность:{round(prob* 100, 2)}%')


    # predicted_number = res_softmax.index(max(res_softmax))
    # print(f'Предсказанная нейросетью цифра {predicted_number}')
    # print(f'Правильная цифра {true_number}')




    """
    ---------------------------------------------------------   BackPropagation       ----------------------------------------------------------------
    Обратное распространение это один из самых базовых способов обучения нейросети. При обратном распространения ошибки происходит следующее
    должно произойти уже прямое распространение для того чтобы обладать уже нужными данными. Дальше после того как мы получили необходимые 
    данные мы отправляем данные обратно на скрытый слой чтобы оно в свою очередь переоформила все ВЕСА, И СМЕЩЕНИЯ дальше это повторяется 
    множества раз (обычно 1.000-10.000) на обучающей выборке с разными картинками чтобы нейросеть не привыкла к одной и тойже картинке 

    """

    # список в котором хранится значения того насколько все нейроны 0 до 9 ошиблись
    delta_output = []

    learning_step = 0.05

    """Вычисляем ошибку на выходе"""

    for i in range(10):
        # если текущее число не является правильным ответом
        if i != true_number:
            # просто добавляем его в список ошибок
            delta_output.append(res_softmax[i]) 
        else:
            # иначе если это правильное число то от данного числа отнимаем 1 
            delta_output.append(res_softmax[i] - 1) 

    """Обновление веса и смещения выходного слоя"""

    # цикл для обработки 10 нейронов выходного слоя (10)
    for j in range(10):
        # цикл для обработки нейронов скрытого слоя (64)
        for i in range(64):
            # вычисляем насколько нейросеть ошиблась и насколько нужно сместиться
            change_weight = delta_output[j] * a_hidden[i]
            # редактируем веса чтобы текущие веса изменились настолько насколько должны изменится изходя из вышеуказанной формулы 
            # и умножаем на шаг обучения 
            weights_output[j][i] -= learning_step * change_weight 
        # обновляем смещения в выходного слоя 
        bias_output[j] -= learning_step * delta_output[j]


    """Считаем ошибку скрытого слоя"""

    # список в котором будет хранится числа которые будут показывать как сильно скрытый слой повлиял на ошибку
    delta_hidden = []

    """--------------------Делаем проверку на работоспособность ReLU так как мы выбрали работать с ReLU-------------------------------------------------------------"""

    # цикл для обработки 10 нейронов выходного слоя (10)
    for i in range(64):
        # переменная в которую будет складываться то на сколько в общем ошиблась нейросеть
        error_sum = 0
        # цикл для обработки нейронов скрытого слоя (64)
        for j in range(10):
            # сумируем все ошибки скрытого слоя
            # Чтобы понять, насколько скрытый нейрон виноват в общей ошибке, мы и суммируем 
            error_sum += delta_output[j] * weights_output[j][i]
        # проверяем текущий элемент больше 0 или нет чтобы понять работает ReLU или произошел какой то сбой
        if z_hidden[i] > 0:
            delta_hidden.append(error_sum)
        else:
            delta_hidden.append(0)

        max_grad = 10

        delta_hidden[i] = max(-max_grad, min(max_grad, delta_hidden[i]))
        
        # обновляем веса смещения скрытого слоя 
        bias_hidden[i] -= learning_step * delta_hidden[i]


    """Обновляем веса и смещения скрытого слоя"""

    # переменная которая будет хранить в себе значения того насколько ошиблась нейросеть
    weight_change = 0

    # цикл для 64 нейронов
    for i in range(64):
        # цикл для выбора одного из 28 списков
        for row in range(28):
            # цикл для выбора одного из 28 списков
            for col in range(28):
                # переменная отвечающая тна то насколько нейросеть ошиблась и насколько ей нужно изменить веса
                weight_change = delta_hidden[i] * img_neurons[row][col]
                # редактируем в чпичке весов весы каждого пикселя вписанного для определенного нейрона с учетом шага обучения
                weights[i][row][col] -= weight_change * learning_step 

    """Сохраняем весь результат обучения нейросети в базе данных (в нашем случае текстовом файле)"""
    # список с правильным ответом
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y[true_number] = 1

    if epoch % 100 == 0:
        # формула расчета кросс-энтропии общепринятая
        # loss = -sum([y[i] * math.log(res_softmax[true_number] + 1e-15) for i in range(10)])
        # та же формула кросс энтропии но только упрощенная
        loss = -math.log(res_softmax[true_number]+ 1e-9)
        total_loss += loss
        correct += (predicted_number == true_number)

        # avg loss - насколько сильно ошибается сеть; accuracy - насколько точная сеть прим. (0.8 = 80%)

        print(f"Epoch {epoch}: avg loss = {total_loss/100:.4f}, accuracy = {correct/100:.2f}")
        total_loss = 0
        correct = 0


        for row in weights_output:
            if any(math.isnan(w) for w in row):
                print("Ошибка: nan в weights_output")
                exit()

        if any(math.isnan(z) or math.isinf(z) for z in z_output):
            print("Ошибка: nan или inf в z_output:", z_output)
            exit()
        
        # сохраняем веса нейронов скрытого слоя 
        with open('db/weight.txt', 'w') as w:
            for i in range(64):
                for row in weights[i]:
                    w.write(' '.join(map(str, row)) + '\n')

        # сохраняем веса нейронов выходного слоя
        with open('db/weight_output.txt', 'w') as w:
            for row in weights_output:
                w.write(' '.join(map(str, row)) + '\n')

        # сохраняем смещения нейронов скрытого слоя
        with open('db/bias_hidden.txt', 'w') as w:
            for b in bias_hidden:
                w.write(str(b) + '\n')

        # сохраняем смещения нейронов выходного слоя
        with open('db/bias_output.txt', 'w') as w:
            for b in bias_output:
                w.write(str(b) + '\n')





