import numpy as np
import matplotlib.pyplot as plt

from common.gradient import numerical_gradient

# 定义梯度下降法的函数
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []
    for i in range(step_num):
        x_history.append(x.copy())
        gradient = numerical_gradient(f, x)
        x -= lr * gradient
    return x, np.array(x_history)

# 定义函数
def function_2(x):
    return x[0]**2 + x[1]**2

if __name__ == '__main__':

    # 定义初始参数
    x = np.array([-3.0, 4.0])

    # 定义超参数
    lr = 0.1
    step_num = 20
    # 使用梯度下降算法，计算最小值点
    descent, history= gradient_descent(function_2, x, lr=lr, step_num=step_num)

    # 绘制图形
    plt.plot([-5, 5], [0, 0], '--b')
    plt.plot([0, 0], [-5, 5], '--b')
    plt.plot(history[:, 0], history[:, 1], 'o')
    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()
    