# 2019/10/21 - Caiye Fan
# XOR Problem by Neural Network
# model: 2 -> 4 -> 1

import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_d(x):
    return x * (1 - x)


def my_sigmoid(x, c):
    return c / (1 + np.exp(-x))


def my_sigmoid_d(x, c):
    return x * (1 - x / c)


def loss(target, y):
    return 1/2 * (target - y)**2


class nn():
    def __init__(self):
        self.lr = 0.1

        # Train Dataset
        self.input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.target = np.array([[0], [1], [1], [0]])

        # Weight 초기화
        self.W0 = np.random.normal(0, 1, size=(2, 4))
        self.W1 = np.random.normal(0, 1, size=(4, 1))
        # self.W0 = np.random.uniform(size=(2, 4))
        # self.W1 = np.random.uniform(size=(4, 1))

        # Bias 초기화
        self.B0 = np.random.normal(0, 1, size=(1, 4))
        self.B1 = np.random.normal(0, 1, size=(1, 1))

        # Activity function 변수 c 초기화
        self.C0 = np.ones((1, 4))
        # self.C0 += 1
        self.C1 = np.ones((1, 1))
        # self.C1 += 1

        self.loss_sum = np.array([])

    # def forward(self):

    # def backward(self):

    def learning(self, epoch):
        for _ in range(epoch):
            # forward
            self.Z1 = np.dot(self.input, self.W0)               # (1,2) x (2,4) = (1,4)
            self.Z1 += self.B0
            # self.X1 = self.sigmoid(self.Z1)  # (1,4)
            self.X1 = my_sigmoid(self.Z1, self.C0)              # (1,4)

            self.Z2 = np.dot(self.X1, self.W1)                  # (1,4) x (4,1) = (1,1)
            self.Z2 += self.B1
            self.y = my_sigmoid(self.Z2, self.C1)               # (1,1)

            self.loss_sum = np.append(self.loss_sum, np.sum(loss(self.target, self.y)))

            # backward
            error_layer_2 = self.target - self.y
            # delta_layer_2 = error_layer_2 * self.sigmoid(self.y)
            delta_layer_2 = error_layer_2 * my_sigmoid_d(self.y, self.C1)

            error_layer_1 = delta_layer_2.dot(self.W1.T)
            delta_layer_1 = error_layer_1 * my_sigmoid_d(self.X1, self.C0)

            # update weights
            self.W1 += self.X1.T.dot(delta_layer_2) * self.lr
            self.W0 += self.input.T.dot(delta_layer_1) * self.lr

            self.B1 += np.sum(delta_layer_2, axis=0, keepdims=True) * self.lr
            self.B0 += np.sum(delta_layer_1, axis=0, keepdims=True) * self.lr

            temp1 = sigmoid(self.Z2) * error_layer_2
            self.C1 += np.sum(temp1, axis=0) * self.lr

            temp0 = sigmoid(self.Z1) * error_layer_1
            # print(sigmoid(self.Z1))
            self.C0 += np.sum(temp0, axis=0) * self.lr

            # self.C0 = self.C1

        print("Activity function c1: ", end='')
        print(*self.C0)
        print("Activity function c2: ", end='')
        print(*self.C1)
        # print(self.loss_sum)

    def draw(self):
        plt.plot(self.loss_sum, '-')
        plt.show()

    def prediction(self, input):
        # forward
        self.Z1 = np.dot(input, self.W0)  # (1,2) x (2,4) = (1,4)
        self.Z1 += self.B0
        self.X1 = my_sigmoid(self.Z1, self.C0)  # (1,4)

        self.Z2 = np.dot(self.X1, self.W1)  # (1,4) x (4,1) = (1,1)
        self.Z2 += self.B1
        y = my_sigmoid(self.Z2, self.C1)  # (1,1)
        return y


if __name__ == "__main__":
    nn = nn()
    nn.learning(5000)
    print(nn.prediction(nn.input))
    nn.draw()