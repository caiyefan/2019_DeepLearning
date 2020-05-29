# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

import numpy as np
from network import Network

# load data
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t_train = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t_test = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])

network = Network(input_size=2, hidden_size=4, output_size=2)

iters_num = 2000
learning_rate = 0.01
# train_size = x_train.shape[0]
# batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []