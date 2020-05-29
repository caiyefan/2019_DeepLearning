from nn_model import *

input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([[0], [1], [1], [0]])
target = to_categorical(target)

model = NN_Model()
model.load_dataset(input, target)
model.add_layer(input_nodes=2, output_nodes=4, activation_type="sigmoid", train_activation=True)
model.add_layer(input_nodes=4, output_nodes=2, activation_type="softmax")
model.learning(epochs=3000, lr=0.1)

model.check(weights=True, activation=True)

model.prediction(input)

