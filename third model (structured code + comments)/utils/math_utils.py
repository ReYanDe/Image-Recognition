"""Подготовка математических утилит"""
import math

"""
Оно возвращает скаларное велечину двух матриц т. е. перемноженную сумму двух матриц друг на друга типо (1*1 + 2*1 + 3*1 + и тд.)
"""

def scalar(matrix1, matrix2):
    res = 0
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            res += matrix1[i][j] * matrix2[i][j]

    return res

def scalar_output_z(a_hidden: list[float], output_weight: list[float]):
    z = 0.0
    for i in range(len(a_hidden)):
        z += a_hidden[i] * output_weight[i]
    
    return z



def softmax(z):
    shift_z = [x - max(z) for x in z]
    exp_scores = [math.exp(i) for i in shift_z]
    total = sum(exp_scores)
    return [x / total for x in exp_scores]