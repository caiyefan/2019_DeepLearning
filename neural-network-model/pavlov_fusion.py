from nn_model import *
plt.style.use('seaborn-whitegrid')

# input data: [food, ring]
# target2: [nothing, sail]
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
target2 = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])

model_1 = NN_Model()
model_1.load_dataset(input, target1)
# model_1.add_layer(input_nodes=2, output_nodes=4, activation_type="sigmoid")
model_1.add_layer(input_nodes=2, output_nodes=4, activation_type="softmax")
model_1.weights = np.array([[[-3.57917628, -4.45802165, 5.51264774, 4.25348689], [-3.49516325, 5.45501402, -4.28458211, 4.33399178]]])
model_1.bias = np.array([[[3.95903399, -0.27024269, -0.41458959, -3.3507687]]])
# model_1.learning(epochs=1000, lr=0.1, regularization=False)


model_2 = NN_Model()
model_2.load_dataset(target1, target2)
model_2.add_layer(input_nodes=4, output_nodes=4, activation_type="sigmoid")
model_2.add_layer(input_nodes=4, output_nodes=2, activation_type="softmax")
model_2.learning(epochs=1000, lr=0.1, regularization=False)

p1 = model_1.prediction(input)
p2 = model_2.prediction(p1)
print("output:")
print(p1)
print(p2)
print("")

model_1.weights = np.array([[[-3.57917628, -4.45802165, 5.51264774, 4.25348689], [-3.49516325, 2.92702158, -0.73537574, 4.33399178]]])
model_1.bias = np.array([[[3.95903399, -2.79823512, 3.13461677, -3.3507687]]])

p1 = model_1.prediction(input)
p2 = model_2.prediction(p1)
print("output:")
print(p1)
print(p2)
print("")

#
# print("")
# print("Result (Before Learning): ")
# model_1.check(weights=True, bias=True)
# p1 = model_1.prediction(input)
# p2 = model_2.prediction(p1)
# print("output:")
# print(p1)
# print(p2)
# print("")
#
# # ----------------------------------------------------------------------------------------
# # Learning Process
# w = np.array([])
# for i in range(50):
#     # model_1.train_dog(input[0], target1[0], lr=0.1, regularization=True)
#     # model_1.train_dog(input[2], target1[2], lr=0.1, regularization=True)
#     # model_1.train_dog(input[3], target1[3], lr=0.1, regularization=True)
#     output = model_1.train_dog(input[1], target1[2], lr=0.1, regularization=False)
#     res = model_2.prediction(output)
#     w = np.append(w, res[0][1])
#     # print(res)
# print()
#
# # Learning Result Print
# print("")
# print("Result (After Learning): ")
# model_1.check(weights=True, bias=True)
# p1 = model_1.prediction(input)
# p2 = model_2.prediction(p1)
# print("output:")
# print(p2)
# print("")
# # ----------------------------------------------------------------------------------------
#
#
# # ----------------------------------------------------------------------------------------
# # Forgetting Process
# for i in range(50):
#     # model_1.train_dog(input[0], target1[0], lr=0.05, regularization=True)
#     # model_1.train_dog(input[2], target1[2], lr=0.05, regularization=True)
#     # model_1.train_dog(input[3], target1[3], lr=0.1, regularization=True)
#     output = model_1.train_dog(input[1], target1[1], lr=0.05, regularization=False)
#     res = model_2.prediction(output)
#     w = np.append(w, res[0][1])
#     # print(res)
#
# # Forgetting Result Print
# print("")
# print("Result (After Forgetting): ")
# model_1.check(weights=True, bias=True)
# p1 = model_1.prediction(input)
# p2 = model_2.prediction(p1)
# print("output:")
# print(p2)
# print("")
# # ----------------------------------------------------------------------------------------
#
# # Result: Probability of Saliva
# plt.plot(w)
# plt.xlabel("Epochs")
# plt.ylabel("Probability(saliva)")
# plt.show()
#
#
#
