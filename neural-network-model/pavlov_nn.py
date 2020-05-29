from nn_model import *

# input data: [ring, food]
# output data1: [forgetting, learning]
# output data2: [nothing, sail]

input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target1 = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])
target2 = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

model_1 = NN_Model()
model_1.load_dataset(input, target1)
model_1.add_layer(input_nodes=2, output_nodes=8, activation_type="sigmoid")
model_1.add_layer(input_nodes=8, output_nodes=2, activation_type="softmax")
model_1.learning(epochs=10000, lr=0.1)

model_2 = NN_Model()
model_2.load_dataset(input, target2)
model_2.add_layer(input_nodes=2, output_nodes=8, activation_type="sigmoid")
model_2.add_layer(input_nodes=8, output_nodes=2, activation_type="softmax")
model_2.learning(epochs=10000, lr=0.1)

model_1.check(weights=True)
model_2.check(weights=True)


model_1.prediction(input)
model_2.prediction(input)

print("-------------------------------------------------")
print("before test:")
model_2.pred(input[2])

w = np.array([])

print("learning:")
for i in range(20):
    res = model_1.train_dog(input[3], target1[3])
    if res == [1]:
        model_2.train_dog(input[2], target2[3])
    w = np.append(w, model_2.pred(input[2])[0][0])

print("forgetting:")
for i in range(20):
    res = model_1.train_dog(input[1], target1[1])
    if res == [0]:
        model_2.train_dog(input[2], target2[2])
    w = np.append(w, model_2.pred(input[2])[0][0])

model_2.check(weights=True)

plt.plot(w)
plt.xlabel("Epochs")
plt.ylabel("Probability(saliva)")
plt.show()
# print("test result:")
# model_2.pred(input[2])