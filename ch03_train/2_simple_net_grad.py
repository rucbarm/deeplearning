import numpy as np

from common.functions import  softmax,cross_entropy_error
from common.gradient import numerical_gradient

# 定义一个简单神经网类
class SimpleNet:
    # 初始化
    def __init__(self):
        self.W = np.random.randn(2, 3)

    # 前向传播
    def forward(self, x):
        a = x @ self.W
        y = softmax(a)
        return  y

    # 损失函数
    def loss(self, x, t):
        y = self.forward(x)
        loss = cross_entropy_error(y, t)
        return loss

# 主流程
net = SimpleNet()

# 输入数据
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

# 前向预测
y = net.forward(x)

# 计算梯度
f= lambda w: net.loss(x, t)
dW = numerical_gradient(f , net.W)
print(dW)