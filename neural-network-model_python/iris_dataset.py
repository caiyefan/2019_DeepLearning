from nn_model import *
from sklearn.datasets import load_iris

data = load_iris()
input = data.data
feature_names = data.feature_names
target = data.target
target_names = data.target_names

input, target = shuffle(input, target)
target = to_categorical(target)
train_x, test_x, train_y, test_y = train_test_split(input, target, test_size=0.2)

model = NN_Model()
model.load_dataset(train_x, train_y, test_x, test_y)
model.add_layer(input_nodes=4, output_nodes=4, activation_type="sigmoid", train_activation=True)
# model.add_layer(input_nodes=16, output_nodes=8, activation_type="sigmoid", train_activation=True)
model.add_layer(input_nodes=4, output_nodes=3, activation_type="softmax")
model.learning(epochs=500, lr=0.01)

model.check()

