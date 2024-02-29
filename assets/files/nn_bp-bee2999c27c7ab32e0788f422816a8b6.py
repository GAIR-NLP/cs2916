import numpy as np

# 初始化参数
np.random.seed(42) # 确保每次运行结果一致
input_size = 2 # 输入层节点数
hidden_size = 3 # 隐藏层节点数
output_size = 2 # 输出层节点数
learning_rate = 0.1 # 学习率

# 随机初始化权重和偏置
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros(output_size)

# ReLU激活函数及其导数
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Softmax函数
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 交叉熵损失及其导数
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred)) / m

def delta_cross_entropy_softmax(y_pred, y_true):
    return y_pred - y_true

# 前向传播
def forward_propagation(X):
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# 反向传播
def backward_propagation(X, Y, Z1, A1, Z2, A2):
    m = X.shape[0]
    
    dZ2 = delta_cross_entropy_softmax(A2, Y)
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0)
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1/m) * np.dot(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0)
    
    return dW1, db1, dW2, db2

# 更新参数
def update_parameters(dW1, db1, dW2, db2):
    global W1, b1, W2, b2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# 示例输入和标签
X = np.array([[0.1, 0.2], [0.3, 0.4]]) # 2个样本
Y = np.array([[1, 0], [0, 1]]) # 对应的one-hot标签

# 执行一次前向传播
Z1, A1, Z2, A2 = forward_propagation(X)

# 计算损失
loss = cross_entropy_loss(A2, Y)

# 执行一次反向传播
dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2)

# 更新参数
update_parameters(dW1, db1, dW2, db2)

print(loss)
