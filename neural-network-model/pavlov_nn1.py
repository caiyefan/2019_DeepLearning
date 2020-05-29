from nn_model import *

# input data: [ring, food]
# output data1: [forgetting, learning]
# output data2: [nothing, sail]

input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([[0, 0], [0, 1], [0, 0], [1, 1]])

model_1 = NN_Model()
model_1.load_dataset(input, target)
model_1.add_layer(input_nodes=2, output_nodes=4, activation_type="sigmoid")
model_1.add_layer(input_nodes=4, output_nodes=2, activation_type="sigmoid")
model_1.learning(epochs=5000, lr=0.1, loss_type="mse")

model_1.check(weights=True)

model_1.prediction(input)


print("-------------------------------------------------")
print("before test:")
model_1.pred(input[2])

w = np.array([])

print("learning:")
for i in range(50):
    res = np.round(model_1.pred(input[3]))
    # res = model_1.train_dog(input[3], target[3])
    if res[0][0] == 1.0:
        model_1.train_dog(input[2], target[1])
    w = np.append(w, model_1.pred(input[2])[0][1])

model_1.check(weights=True)


print("forgetting:")
for i in range(50):
    res = np.round(model_1.pred(input[1]))
    # res = model_1.train_dog(input[1], target[1])
    if res[0][0] == 0.0:
        model_1.train_dog(input[2], target[2])
    w = np.append(w, model_1.pred(input[2])[0][1])

model_1.prediction(input)

# model_1.check(weights=True)

plt.plot(w)
plt.xlabel("Epochs")
plt.ylabel("Probability(saliva)")
plt.show()