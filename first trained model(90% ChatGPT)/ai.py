import numpy as np
import cv2
import random
import math

from random_choice import random_img


# a = [[round(random.uniform(-1, 1), 1) for i in range(28)] for j in range(28)]
# with open('db/weight.txt', 'a') as w:
#     for i in a:
#         for j in i:
#             w.write(str(j) + " ")
#         w.write("\n")
# weights = [
#     [ [round(random.uniform(-1, 1), 4) for _ in range(28)] for _ in range(28) ]
#     for _ in range(64)
# ]

# with open('db/weight.txt', 'w') as f:
#     for i in range(64):
#         for row in weights[i]:
#             f.write(' '.join(map(str, row)) + '\n')

# with open('db/weight_output.txt', 'w')as w:
#     for _ in range(10):
#         line = ' '.join([str(round(random.uniform(-1, 1), 1)) for _ in range(64)])
#         w.write(line + '\n')


 # призыв чисел из файла обработчика
img1, firstN = random_img()
# print(img1[1])
# print(len(img1), len(img1[0]))


output_biases = []
with open('db/bias_output.txt', 'r') as f:
    output_biases = (list(map(float, f.readline().split())))

output_weights = []
with open('db/weight_output.txt', 'r') as f:
    for line in f.readlines():
        output_weights.append(list(map(float, line.split())))

weights = []
with open('db/weight.txt', 'r') as w:
    p = [ list(map(float,i.split())) for i in w.readlines()] 

weights = [p[i*28:(i+1)*28]  for i in range(64)]

total_loss = 0
correct = 0

# начинаю эпоху обучения 
for epoch in range(10000):

    

   

    def scalar_product(matrix1, matrix2):
        result = 0
        for i in range(len(matrix1)):
            for j in range(len(matrix1[0])):
                result += matrix1[i][j] * matrix2[i][j]
        return result

    def scalar_product_vector(v1, v2):
        return sum([v1[i] * v2[i] for i in range(len(v1))])


    def softmax(z_list):
        exps = [math.exp(z) for z in z_list]
        sum_exps = sum(exps)
        return [e / sum_exps for e in exps]


    neuron = []

        
    for i in range(64):
        w_i = weights[i]
        b = 0.0
        z = scalar_product(img1, w_i) + b
        ReLU = max(0, z)
        neuron.append(ReLU)


    # Выходной слой


    z_output = []

    for i in range(10):
        z = scalar_product_vector(neuron, output_weights[i]) + output_biases[i]
        z_output.append(z)

    output_probs = softmax(z_output)

    predicted_digit = output_probs.index(max(output_probs))

    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y[firstN] = 1
    delta2 = []

    if epoch % 100 == 0:

        loss = -sum([y[i] * math.log(output_probs[i] + 1e-15) for i in range(10)])
        total_loss += loss
        correct += (predicted_digit == firstN)

        # avg loss - насколько сильно ошибается сеть; accuracy - насколько точная сеть прим. (0.8 = 80%)

        print(f"Epoch {epoch}: avg loss = {total_loss/100:.4f}, accuracy = {correct/100:.2f}")


        with open('db/weight.txt', 'w') as w:
            for i in range(64):
                for row in weights[i]:  # Каждая строка 28 чисел
                    line = ' '.join([str(round(val, 4)) for val in row])
                    w.write(line + '\n')


        with open('db/weight_output.txt', 'w') as w:
            for i in range(10):
                # for row in output_weights[i]:  # Каждая строка 28 чисел
                line = ' '.join([str(round(val, 4)) for val in output_weights[i]])
                w.write(line + '\n')

        with open('db/bias_output.txt', 'w') as w:
            line = ' '.join([str(round(i, 4)) for i in output_biases])
            w.write(line + '\n')


    # for i in range(len(output_probs)):

    #     print("Выходные вероятности:", output_probs[i])
    # print("Распознанная цифра:", predicted_digit)
    # print('Правильное число ', firstN)



    for j in range(10):
        delta2.append(output_probs[j] - y[j])

    dReLU = []

    for i in range(len(neuron)):
        if neuron[i] > 0:
            dReLU.append(1)
        else: 
            dReLU.append(0)


    delta1 = []

    for i in range(len(neuron)):
        sum_error = 0
        for j in range(10):
            sum_error += delta2[j] * output_weights[j][i]
        
        delta1.append(sum_error * dReLU[i])

    learning_rate = 0.075

    for j in range(10):
        for i in range(64):
            output_weights[j][i] -= learning_rate * neuron[i] * delta2[j] 
        output_biases[j] -= learning_rate * delta2[j]

    for i in range(64):
        for m in range(28):
            for n in range(28):
                weights[i][m][n] -= learning_rate * img1[m][n] * delta1[i]
            

    

