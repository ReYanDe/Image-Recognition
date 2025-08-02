import math

"""
Оно возвращает скаларное велечину двух матриц т. е. перемноженную сумму двух матриц друг на друга типо (1*1 + 2*1 + 1*3 + и тд.)
"""
def scalar(matrix1, matrix2):
    res = 0
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            res += matrix1[i][j] * matrix2[i][j]

    return res


def softmax(z_list):
    max_z = max(z_list)
    exps = [math.exp(z - max_z) for z in z_list]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]
