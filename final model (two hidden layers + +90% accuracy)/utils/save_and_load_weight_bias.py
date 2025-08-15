"""Загрузка и сохранение данных для базы данных"""
import random


# # Создается веса для 64 нейронов сразу в переменную weights
# weights = [[[round(random.uniform(-1, 1) , 4) for _ in range(28)] for _ in range(28)] for _ in range(64)]

"""Создание рандомных чисел для матрицы весов состоящая из 64 нейронов"""

# with open('./weights/hidden_weights.txt', 'w') as w:
#     # 64 раза проходимся по циклу для 64 нейронов
#     for i in range(64):
#         # проверяем каждый элемент в матрице
#         for row in weights[i]:
#             # переобразуем список в простые числа, добавляем пробелы для каждой матрицы
#             w.write(' '.join(map(str, row)) + '\n')



# # Создается веса для 32 нейронов второго скрытого слоя
# weights_2 = [[round(random.uniform(-1, 1) , 4) for _ in range(64)] for _ in range(32)]

"""Создание матрицы весов размером 64х32 (32 нейрона второго скрытого слоя с 64 нейронами первого скрытого слоя )"""

# with open('./weights/hidden_weights_2.txt', 'w') as w:
#     # проверяем каждый список индивидуально
#     for row in weights_2:
#         # вписываем каждое число без запятых только по пробелу и в конце вставляем отступ
#         w.write(' '.join(map(str, row)) + '\n')



"""Создание рандомных чисел для матрицы весов состоящая из 64 готовых ответа и 10 нейронов выходного слоя"""

# # создание рандомных чисел для весов 10х64 
# weights_output = [[round(random.uniform(-1, 1) , 4) for _ in range(32)] for _ in range(10)]


# with open('./weights/output_weights.txt', 'w') as w:
#     # проверяем каждый список индивидуально
#     for row in weights_output:
#         # вписываем каждое число без запятых только по пробелу и в конце вставляем отступ
#         w.write(' '.join(map(str, row)) + '\n')

# """Создания смещений (bias) для скрытого слоя (64 нейрона)"""


# # создание рандомных чисел для смещений (64) 
# bias_hidden = [round(random.uniform(-0.1, 0.1) , 4) for _ in range(64)]


# with open('./weights/bias_hidden.txt', 'w') as w:
#     # проверяем каждый список индивидуально
#     for b in bias_hidden:
#         # вписываем каждое число без запятых только по пробелу и в конце вставляем отступ
#         w.write(str(b) + '\n')

"""Создания смещений (bias) для скрытого слоя (10 нейрона)"""


# # создание рандомных чисел для смещений (10) 
# bias_output = [round(random.uniform(-0.1, 0.1) , 4) for _ in range(10)]


# with open('./weights/bias_output.txt', 'w') as w:
#     # проверяем каждый список индивидуально
#     for b in bias_output:
#         # вписываем каждое число без запятых только по пробелу и в конце вставляем отступ
#         w.write(str(b) + '\n')


"""Создания смещений (bias) для второго скрытого слоя (32 нейрона)"""


# # создание рандомных чисел для смещений (10) 
# bias_hidden_2 = [round(random.uniform(-0.1, 0.1) , 4) for _ in range(32)]


# with open('./weights/bias_hidden_2.txt', 'w') as w:
#     # проверяем каждый список индивидуально
#     for b in bias_hidden_2:
#         # вписываем каждое число без запятых только по пробелу и в конце вставляем отступ
#         w.write(str(b) + '\n')



"""Чтение данных из базы данных (текстовый файл)"""


# # Открытие матрицы весов состоящая из 64 нейронов



weights = []
def read_weight_hidden_def():
    weights = []
    with open('./weights/hidden_weights.txt', 'r') as r:
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
        return weights


# Чтение веса выходного слоя 10 нейронов 


# weights_output = []
def read_output_weight_def():
    weights_output = []
    with open('./weights/output_weights.txt', 'r') as r:
        # проходимся по списку элементов 10 нейронов
        for i, line in enumerate(r):
            # очищаем от лишних пробелов и превращаем в список
            row = list(map(float, line.strip().split()))
            weights_output.append(row)

        return weights_output



# Чтение второй скрытый слой нейросети 

# weights_output = []
def read_second_hidden_weight_def():
    weights_2_hidden = []
    with open('./weights/hidden_weights_2.txt', 'r') as r:
        # проходимся по списку элементов 10 нейронов
        for i, line in enumerate(r):
            # очищаем от лишних пробелов и превращаем в список
            row = list(map(float, line.strip().split()))
            weights_2_hidden.append(row)

        return weights_2_hidden





# Чтение смещений (bias) входного слоя 64 нейронов 


# bias_hidden = []
def read_bias_hidden_def():
    bias_hidden = []
    with open('./weights/bias_hidden.txt', 'r') as r:
        # проходимся по списку элементов 64 нейронов
        for i, line in enumerate(r):
            # очищаем от лишних пробелов и превращаем в список
            bias_hidden.append(float(line.strip()))
        return bias_hidden


#Чтение смещений (bias) входного слоя 64 нейронов 

# bias_output = []
def read_bias_output_def():
    bias_output = []
    with open('./weights/bias_output.txt', 'r') as r:
        # проходимся по списку элементов 10 нейронов
        for i, line in enumerate(r):
            # очищаем от лишних пробелов и превращаем в список
            bias_output.append(float(line.strip()))
        return bias_output
    

# чтения биаса второго скрытого слоя

def read_bias_second_def():
    bias_second_weight = []
    with open('./weights/bias_hidden_2.txt', 'r') as r:
        # проходимся по списку элементов 10 нейронов
        for i, line in enumerate(r):
            # очищаем от лишних пробелов и превращаем в список
            bias_second_weight.append(float(line.strip()))
        return bias_second_weight
    




"""Записывание изменения данных в базу данных после каждой эпохи (текстовый файл)"""



def save_weights_hidden_def(weights):
# сохраняем веса нейронов скрытого слоя 
    with open('./weights/hidden_weights.txt', 'w') as w:
        for i in range(64):
            for row in weights[i]:
                w.write(' '.join(map(str, row)) + '\n')


def save_weights_hidden_2_def(weights_hidden_2 ):
    # сохраняем веса нейронов выходного слоя
    with open('./weights/hidden_weights_2.txt', 'w') as w:
        for row in weights_hidden_2 :
            w.write(' '.join(map(str, row)) + '\n')

        
def save_weights_output_def(weights_output):
    # сохраняем веса нейронов выходного слоя
    with open('./weights/output_weights.txt', 'w') as w:
        for row in weights_output:
            w.write(' '.join(map(str, row)) + '\n')


def save_bias_hidden_def(bias_hidden):
    # сохраняем смещения нейронов скрытого слоя
    with open('./weights/bias_hidden.txt', 'w') as w:
        for b in bias_hidden:
            w.write(str(b) + '\n')


def save_bias_hidden_2_def(bias_hidden_2):
    # сохраняем смещения нейронов выходного слоя
    with open('./weights/bias_hidden_2.txt', 'w') as w:
        for b in bias_hidden_2:
            w.write(str(b) + '\n')


def save_bias_output_def(bias_output):
    # сохраняем смещения нейронов выходного слоя
    with open('./weights/bias_output.txt', 'w') as w:
        for b in bias_output:
            w.write(str(b) + '\n')