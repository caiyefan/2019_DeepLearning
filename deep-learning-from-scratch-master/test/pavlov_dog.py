# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

import numpy as np
from network import MyNetwork
from matplotlib import pyplot as plt

# load data
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t_train = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t_test = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

network = MyNetwork(input_size=2, hidden_size=8, output_size=2)

iters_num = 5000
learning_rate = 0.1

train_size = x_train.shape[0]
batch_size = 4
iter_per_epoch = max(train_size / batch_size, 1)

train_loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(iters_num):
    # batch_mask = np.random.choice(train_size, batch_size)
    # x_batch = x_train[batch_mask]
    # t_batch = t_train[batch_mask]
    x_batch = x_train
    t_batch = t_train

    # Calculate Gradient
    grad = network.gradient(x_batch, t_batch)

     # Update Gradient
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        train_acc_list.append(train_acc)
        test_acc = network.accuracy(x_test, t_test)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

print(network.predict(x_train))

plt.style.use('seaborn-whitegrid')
plt.plot(train_loss_list, label='Loss')
plt.title("Learning Result")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.show()