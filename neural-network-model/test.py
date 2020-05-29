from nn_model import *

# input data [food, bell]
# input = np.array([[0, 0], [1, 0]])
input = np.array([[0, 0], [0, 1], [1, 0]])
target = np.array([[0], [0], [1]])
target = to_categorical(target)
test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

model = NN_Model()
model.load_dataset(input, target)
# model.add_layer(input_nodes=2, output_nodes=2, activation_type="sigmoid")
model.add_layer(input_nodes=2, output_nodes=2, activation_type="softmax")
model.learning(epochs=1000, lr=0.1, loss_type="mse")

model.check(weights=True)
model.prediction(test)

# input2 = np.array([[0, 0], [1, 1]])
# target2 = np.array([[0], [1]])
input2 = np.array([[0, 0], [1, 0], [0, 1]])
target2 = np.array([[0], [1], [1]])
target2 = to_categorical(target2)
model.load_dataset(input2, target2)
model.learning(epochs=100, lr=0.1, loss_type="mse")

model.check(weights=True)
model.prediction(test)
