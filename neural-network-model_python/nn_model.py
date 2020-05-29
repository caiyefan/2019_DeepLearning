# 2019/10/21 - Caiye Fan
# Neural Network Model

from functions import *


class NN_Model():
    def __init__(self):
        # 初始化数据集
        self.train_x = []  # 训练数据-输入
        self.train_y = []  # 训练数据-目标
        self.test_x = []  # 测试数据-输入
        self.test_y = []  # 测试数据-目标

        # 初始化储存权重，偏差等变量所需的list
        self.weights = []  # 储存每层权重
        self.bias = []  # 储存每层偏差
        self.activation_a = [[1]]  # 储存激活函数变量
        self.activation_b = [[1]]  # 储存激活函数变量
        self.activation_type = ["none"]  # 储存每层激活函数类型
        self.train_activation = [False]  # 是否训练该层激活函数变量

        # 初始化储存神经网络计算所需的变量的list
        self.Z = []  # Z=WX+B
        self.X = []  # X=f(Z), f:激活函数
        self.error_layer = [0]  # 反向传播计算时所需的层误差
        self.delta_layer = [0]  # 反向传播计算时所需的层梯度

        self.num_layer = 0  # layer层数
        self.loss_list = np.array([])  # 储存每次训练的损失
        self.acc_list = np.array([])  # 储存每次训练的准确率
        self.w_list1 = np.array([])  # 储存权重的变化
        self.w_list2 = np.array([])  # 储存权重的变化
        self.cost_time = 0  # 所用的时间
        # self.loss_1 = np.array([])
        # self.loss_2 = np.array([])
        # self.loss_3 = np.array([])
        # self.loss_4 = np.array([])

    def load_dataset(self, train_x, train_y, test_x=None, test_y=None):
        # 载入数据集
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.num_inputnodes = len(self.train_x[0])  # 数据输入的大小
        self.num_outnodes = len(self.train_y[0])  # 数据输出的大小

        # 设置输入层的大小(神经元数)
        self.Z.append(self.train_x[0].reshape(1, self.num_inputnodes))
        self.X.append(self.train_x[0].reshape(1, self.num_inputnodes))

        # print数据集的大小
        print("Dataset Size: ", end=' ')
        print("train data:", len(train_x), end=', ')
        if test_x is not None:
            print("test data:", len(test_x))
        else:
            print("test data: 0")

    def add_layer(self, input_nodes, output_nodes, activation_type="sigmoid", train_activation=False):
        # 往神经网络模型里添加层时，往各list里初始化相应大小的变量
        self.Z.append(np.zeros((1, output_nodes)))
        self.X.append(np.zeros((1, output_nodes)))
        self.weights.append(np.random.normal(0, 1, (input_nodes, output_nodes)))
        self.bias.append(np.random.normal(0, 1, (1, output_nodes)))
        self.activation_type.append(activation_type)
        self.activation_a.append(np.ones((1, output_nodes)))
        self.activation_b.append(np.ones((1, output_nodes)))

        self.train_activation.append(train_activation)
        self.error_layer.append(0)
        self.delta_layer.append(0)
        self.num_layer += 1

    def forward(self, input):
        # 前馈神经网络
        # 数据从输入到输出每一层的计算
        # Z[1] = W * X[0] + B
        # X[1] = f(Z[1]), f:activation function
        self.X[0] = input.reshape(1, self.num_inputnodes)
        for i in range(self.num_layer):
            self.Z[i + 1] = np.dot(self.X[i], self.weights[i])
            self.Z[i + 1] += self.bias[i]
            self.X[i + 1] = activation_function(self.Z[i + 1], self.activation_a[i + 1],
                                                self.activation_b[i + 1],
                                                type=self.activation_type[i + 1], )
        return self.X[self.num_layer]

    def back_propagation(self, target, loss_type):
        # 反向传播
        # 从输出层开始，反向计算每一层的误差和梯度
        for i in range(self.num_layer):
            j = self.num_layer - i
            if j == self.num_layer:
                if self.activation_type[j] == "softmax":
                    # self.error_layer[j] = loss_function_d(target, self.X[j], type=loss_type)
                    # self.delta_layer[j] = target - self.X[j]
                    self.delta_layer[j] = self.X[j] - target
                else:
                    self.error_layer[j] = loss_function_d(target, self.X[j], type=loss_type)
                    self.delta_layer[j] = self.error_layer[j] * activation_function_d(self.X[j], self.activation_a[j],
                                                                                      self.activation_b[j],
                                                                                      type=self.activation_type[j])
            else:
                self.error_layer[j] = self.delta_layer[j + 1].dot(self.weights[j].T)
                self.delta_layer[j] = self.error_layer[j] * activation_function_d(self.X[j], self.activation_a[j],
                                                                                  self.activation_b[j],
                                                                                  type=self.activation_type[j])

    def update(self, lr=0.01):
        # 更新每层的权重，偏差，激活函数变量
        for i in range(self.num_layer):
            j = self.num_layer - i

            self.weights[j - 1] -= (self.X[j - 1].T.dot(self.delta_layer[j])) * lr
            # self.weights[j - 1] -= (self.X[j - 1].T.dot(self.delta_layer[j]) + 0.01 * self.weights[j - 1]) * lr

            self.bias[j - 1] -= np.sum(self.delta_layer[j], axis=0, keepdims=True) * lr

            # Update activation function variable c
            if self.activation_type[j] == "sigmoid" and self.train_activation[j]:
                self.activation_a[j] -= np.sum(self.Z[j]*sigmoid(self.Z[j], self.activation_a[j], self.activation_b[j])*(1 - sigmoid(self.Z[j], self.activation_a[j], 1)) * self.error_layer[j], axis=0) * lr
                self.activation_b[j] -= np.sum(sigmoid(self.Z[j], self.activation_a[j], 1) * self.error_layer[j], axis=0) * lr

    def learning(self, epochs=1, lr=0.1, loss_type="cross entropy"):
        begin_time = time()
        for e in range(epochs):
            local_time = time()
            right = 0.0
            loss_sum = 0

            for i in range(len(self.train_y)):
                input = self.train_x[i]
                target = self.train_y[i]

                res = self.forward(input)
                self.back_propagation(target, loss_type)
                self.update(lr)

                loss_sum += loss_function(target, res, loss_type)

                r = np.round(res)
                # r = props_to_onehot(res)
                if (r == target).all():
                    right += 1

                # Iris Dataset
                # if (target == [1., 0., 0.]).all():
                #     loss_1.append(my_loss(target, res))
                # elif (target == [0., 1., 0.]).all():
                #     loss_2.append(my_loss(target, res))
                # elif (target == [0., 0., 1.]).all():
                #     loss_3.append(my_loss(target, res))

                # XOR Problem
                # if (input == self.train_x[0]).all():
                #     self.loss_1 = np.append(self.loss_1, my_loss(target, res))
                # elif (input == self.train_x[1]).all():
                #     self.loss_2 = np.append(self.loss_2, my_loss(target, res))
                # elif (input == self.train_x[2]).all():
                #     self.loss_3 = np.append(self.loss_3, my_loss(target, res))
                # elif (input == self.train_x[3]).all():
                #     self.loss_4 = np.append(self.loss_4, my_loss(target, res))

            # Iris Dataset
            # self.loss_1 = np.append(self.loss_1, np.mean(loss_1))
            # self.loss_2 = np.append(self.loss_2, np.mean(loss_2))
            # self.loss_3 = np.append(self.loss_3, np.mean(loss_3))
            # self.loss = np.append(self.loss, loss_sum / len(self.train_y))

            w0 = np.array(self.weights[0])
            w = np.array([])
            for i in range(len(w0)):
                w = np.append(w, w0[i].flatten())
            w = w.flatten()
            if self.w_list1.shape[0] == 0:
                self.w_list1 = np.append(self.w_list1, w)
            else:
                self.w_list1 = np.column_stack((self.w_list1, w))

            w1 = np.array(self.weights[1])
            w = np.array([])
            for i in range(len(w1)):
                w = np.append(w, w1[i].flatten())
            w = w.flatten()
            if self.w_list2.shape[0] == 0:
                self.w_list2 = np.append(self.w_list2, w)
            else:
                self.w_list2 = np.column_stack((self.w_list2, w))


            loss_sum /= len(self.train_y)
            self.loss_list = np.append(self.loss_list, loss_sum)
            acc = round(right / len(self.train_y) * 100, 2)
            self.acc_list = np.append(self.acc_list, acc)
            self.cost_time = round((time() - local_time) * 1000, 2)

            print("[Epoch", e + 1, "/", epochs, end="]")
            print(" -Loss:", round(float(loss_sum), 2), end=",")
            print(" -Accuracy:", acc, end="%")
            print(" -Time:", self.cost_time, end="ms")
            print('')

        self.cost_time = (time() - begin_time) * 1000

    def train_dog(self, input, target, lr=0.1, loss_type="mse"):
        right = 0.0
        loss_sum = 0
        r = 0

        res = self.forward(input)
        self.back_propagation(target, loss_type)
        self.update(lr)

        w0 = np.array(self.weights[0])
        w = np.array([])
        for i in range(len(w0)):
            w = np.append(w, w0[i].flatten())
        w = w.flatten()
        if self.w_list1.shape[0] == 0:
            self.w_list1 = np.append(self.w_list1, w)
        else:
            self.w_list1 = np.column_stack((self.w_list1, w))

        w1 = np.array(self.weights[1])
        w = np.array([])
        for i in range(len(w1)):
            w = np.append(w, w1[i].flatten())
        w = w.flatten()
        if self.w_list2.shape[0] == 0:
            self.w_list2 = np.append(self.w_list2, w)
        else:
            self.w_list2 = np.column_stack((self.w_list2, w))

        loss_sum += loss_function(target, res, loss_type)
        # r = props_to_onehot(res)
        if (r == target).all():
            right += 1

        # r = np.argmax(r, axis=1)
        r = np.round(res)
        print(res)
        # print(r)
        return r

    def model_score(self, test_x=None, test_y=None):
        if test_x is None:
            test_x = self.test_x
            test_y = self.test_y

        if test_x is None:
            print("No Test Data!")
            return

        res = np.array([])
        for i in range(len(test_x)):
            if not len(res) > 0:
                res = self.forward(test_x[i])
            else:
                res = np.r_[res, self.forward(test_x[i])]
        res = props_to_onehot(res)
        res = np.argmax(res, axis=1)
        test_y = np.argmax(test_y, axis=1)

        print("Test Dataset Size: ", len(test_y))

        loss = round(loss_function(test_y, res)/len(test_y), 2)
        print("Test Loss: ", loss)

        acc = round(metrics.accuracy_score(test_y, res) * 100, 2)
        print("Test Accuracy: ", acc, "%")

        # recal = metrics.recall_score(test_y, res)
        # print("Test Recal: ", recal)

    def prediction(self, input):
        res = np.array([])
        for i in range(len(input)):
            if not len(res) > 0:
                res = self.forward(input[i])
            else:
                res = np.r_[res, self.forward(input[i])]
        r = np.round(res)
        res_1 = props_to_onehot(res)
        res_1 = np.argmax(res_1, axis=1)

        print(r)
        print(res)
        # print(res_1)
        return res

    def pred(self, input):
        res = np.array([])
        res = self.forward(input)
        r = np.round(res)
        # res_1 = props_to_onehot(res)
        # res_1 = np.argmax(res_1, axis=1)
        # print(res)
        return res

    def draw(self):
        plt.style.use('seaborn-whitegrid')

        plt.subplot(211)
        plt.plot(self.loss_list, label='Loss')
        plt.title("Learning Result")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.subplot(212)
        plt.plot(self.acc_list, label="Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.show()

        plt.subplot(211)
        for i in range(len(self.w_list1)):
            plt.plot(self.w_list1[i])
        plt.xlabel("Epochs")
        plt.ylabel("Weight")

        plt.subplot(212)
        for i in range(len(self.w_list2)):
            plt.plot(self.w_list2[i])
        plt.xlabel("Epochs")
        plt.ylabel("Weight")

        plt.show()

        # plt.plot(range(len(self.loss_1)), self.loss_1, '-', label='setosa')
        # plt.plot(range(len(self.loss_2)), self.loss_2, '-', label='versicolor')
        # plt.plot(range(len(self.loss_3)), self.loss_3, '-', label='virginica')

        # plt.plot(range(len(self.loss_1)), self.loss_1 * (-1), '-', label='[0,0]')
        # plt.plot(range(len(self.loss_2)), self.loss_2, '-', label='[0,1]')
        # plt.plot(range(len(self.loss_3)), self.loss_3, '-', label='[1,0]')
        # plt.plot(range(len(self.loss_4)), self.loss_4 * (-1), '-', label='[1,1]')
        #
        # plt.title("Learning Result")
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.show()

    def check(self, weights=None, bias=None, activation=None):
        print("")
        print("[Trainning Result]")
        print("Number of Layer: ", end='')
        print(self.num_layer, "layers")

        if weights:
            print("Weights: ", end='')
            print(*self.weights)

        if bias:
            print("Bias: ", end='')
            print(*self.bias)

        if activation:
            print("Activation type: ", end='')
            print(*self.activation_type)
            print("Activation_a: ", end='')
            print(*self.activation_a)
            print("Activation_b: ", end='')
            print(*self.activation_b)

        print("Cost Time: ", end='')
        print(round(self.cost_time, 2), "ms")

        # print("Z: ", end='')
        # print(*self.Z)
        # print("X: ", end='')
        # print(*self.X)
        # print("error_layer: ", end='')
        # print(*self.error_layer)
        # print("delta_layer: ", end='')
        # print(*self.delta_layer)

        self.model_score()
        self.draw()
