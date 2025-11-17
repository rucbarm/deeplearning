import  numpy as np
import matplotlib.pyplot as plt
from common.gradient import numerical_diff
# 原函数 y = 0.01x^2 + 0.1x
def func(x):
    return 0.01 * x ** 2 + 0.1 * x

# 切线方程函数，返回切线函数
def tangent_line(f, x):
    y = func(x)

    # 利用数值微分
    a = numerical_diff(f, x)
    print(a)
    b = y - a * x
    return lambda t: a * t + b

# 绘制图像
# 定义画图范围
x = np.arange(0.0, 10.0, 0.1)
y = func(x) #原函数

tangent = tangent_line(func, x=5.0)
print(tangent(5.0))
f_line = tangent(x)

# 绘制图像
plt.plot(x,y) #原函数
plt.plot(x,f_line)#切点为5的切线函数
plt.show()
