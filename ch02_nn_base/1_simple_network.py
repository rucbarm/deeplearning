import numpy as np
from common.functions import sigmoid,identity

# 初始化神经网络
def init_network():
    network = {}
    #第一层参数
    network['w1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    #第二层参数
    network['w2'] = np.array([[0.2,0.6],[0.5,0.1],[0.3,0.4]])
    network['b2'] = np.array([0.9,0.3])
    #第三层参数
    network['w3'] = np.array([[0.2,0.5],[0.4,0.6]])
    network['b3'] = np.array([0.6,0.1])
    return network

# 前向传播
def forward(network,x):
    w1,w2,w3 = network['w1'],network['w2'],network['w3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    # 第一层
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)

    # 第二层
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)

    # 第三层
    a3 = np.dot(z2, w3) + b3
    y = identity(a3)
    return y

# 测试主流程
network  = init_network()

x = np.array([1.0,0.5])

# 前向传播的过程就是预测过程
y = forward(network,x)
print(y)
