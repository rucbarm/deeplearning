# 数值微分求导函数

def numerical_diff(f, x):
    h = 1e-4 # 1e-4
    return  (f(x+h) - f(x-h)) / (2*h)