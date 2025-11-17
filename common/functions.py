# 阶跃函数
def step_function0(x):
    if x > 0:
        return 1
    else:
        return 0

import numpy as np

def step_function1(x):
    return np.array(x > 0, dtype=int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

def identity(x):
    return x

# 损失函数
def mean_sqaured_error(y, t):
    return 0.5 * np.sum(np.power(y - t, 2))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        t = t.argmax(axis=1)
    return -np.sum(np.log(y[np.arange(t.size), t] + 1e-10))
if __name__ == '__main__':
    x = np.array([0,1,2,3,4,5,-1,0,1,2,3,4,5])
    print(step_function1(x))

