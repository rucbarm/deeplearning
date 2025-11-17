import numpy as np
from common.functions import softmax, sigmoid, identity
import matplotlib.pyplot as plt
import  pandas as pd
import  joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 读取数据
def get_data():
    # 1.从文件加载数据集
    data = pd.read_csv("../data/train.csv")

    # 2.将数据集拆分为训练集和测试集
    x = data.drop(["label"], axis=1)
    y = data["label"]

    x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.3, random_state=42)

    # 3.归一化处理
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    scaler.transform(x_test)

    return x_test , y_test

def init_network():

    # 1.加载模型
    network = joblib.load("../data/nn_sample")

    return network


def forward(network,x):
    w1,w2,w3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    # 第一层
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)

    # 第二层
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)

    # 第三层
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y

x_test,y_test = get_data()

# 2. 创建模型（加载参数）
network = init_network()

# 3. 前向传播（预测）
y_proba = forward(network, x_test)
print(y_proba.shape)

# 4. 将分类概率转换为分类标签
# argmax()函数返回的是最大值的索引，而不是最大值本身
y_pred = np.argmax(y_proba, axis=1)
print(y_pred.shape)
# print(x_test.shape)
# print(y_test.shape)
print(len(y_test))
# 5. 计算准确率
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(accuracy)
