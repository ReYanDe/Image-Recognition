import matplotlib.pyplot as plt
import numpy as np
import random as r

# x = [1, 2, 3, 4, 5]
# y = [4, 12, 36, 25, 22]

# x1 = [1, 2, 3, 4, 5]
# y1 = [4, 1322, 36, 235, 221]



# функция для создания одного графика
def graf(a:list, b:list):
    # создание графика
    plt.plot(a, b, marker='o', linestyle='-', color='b')

    # отображение сетки
    plt.grid(True)

    # отображение графика
    plt.show()



# функция для графика в котором несколько линий в одном графике
def graf_lines(a:list, b:list):


    for  i in range(10):
        plt.plot(a, b, label=f'График {i}')

    plt.title('10 графиков на одной плоскости')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.legend()
    plt.grid(True)
    plt.show()


# x = np.linspace(0, 10, 100)





# создание несколько графиков  
def some_grafs(a:list, b:list):

    fig, axes = plt.subplots(5, 2, figsize=(5, 15), constrained_layout=True)


    for i, ax in enumerate(axes.flat):
        ax.plot(a[i], b[i], marker='o')
        ax.grid(True)

    plt.show()

# предопределенные значения для графиков
x = [[1, 2, 3, 5], [3, 4], [5,6] ,[7 , 8], [1, 10],[1, 2], [3, 4], [5,6] ,[7 , 8], [1, 10],]
y = [[-2, -5, -3, -5], [1, 4],  [5,6] ,[7 , 8], [1, 10], [1, 2], [3, 4], [5,6] ,[7 , 8], [1, 10],]

# рандом для графиков
data = [[r.randint(-10, 20) for _ in range(5)] for _ in range(10)]
data2 = [[r.randint(-10, 20) for _ in range(5)] for _ in range(10)]

# просто удобна складывает 
m = np.array(data)

# print(data)

some_grafs(data, data2)

