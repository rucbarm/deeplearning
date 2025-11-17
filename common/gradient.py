import numpy as np

# 数值微分求导函数
def numerical_diff(f, x):
    h = 1e-4 # 1e-4
    return  (f(x+h) - f(x-h)) / (2*h)

# 数值微分求梯度，f为多元函数，x为向量，一组参数
def _numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # 生成和x形状相同的数组
    for i in range(x.size):
        tmp_val = x[i]
        x[i] = tmp_val + h
        fxh1 = f(x)
        x[i] = tmp_val - h
        fxh2 = f(x)
        grad[i] = (fxh1 - fxh2) / (2*h)
        x[i] = tmp_val
    return grad

# 输入X为矩阵，多组参数
def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient(f, X)
    else:
        grad = np.zeros_like(X)
        for i,x in enumerate(X):
            grad[i] = _numerical_gradient(f, x)
        return grad


