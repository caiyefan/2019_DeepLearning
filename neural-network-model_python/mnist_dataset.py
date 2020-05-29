from nn_model import *
from keras.datasets import mnist


(train_x, train_y), (test_x, test_y) = mnist.load_data()

# dataset pre-processing
train_x = train_x.reshape(train_x.shape[0], 784)
test_x = test_x.reshape(test_x.shape[0], 784)
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

train_x, train_y = shuffle(train_x, train_y)
test_x, test_y = shuffle(test_x, test_y)

train_x = train_x / 255
test_x = test_x / 255


model = NN_Model()
model.load_dataset(train_x, train_y, test_x, test_y)
model.add_layer(input_nodes=784, output_nodes=128, activation_type="sigmoid", train_activation=True)
model.add_layer(input_nodes=128, output_nodes=10, activation_type="softmax")
model.learning(epochs=15, lr=0.01)

model.check()