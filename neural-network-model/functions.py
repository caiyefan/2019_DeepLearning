import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn import metrics
from time import *
from tqdm import tqdm
import time

def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b


def activation_function(x, a=1, b=1, type="sigmoid"):
    if type == "sigmoid":
        return sigmoid(x, a, b)
    elif type == "softmax":
        return softmax(x)
    elif type == "relu":
        return relu(x)
    elif type == "tanh":
        return tanh(x)


def activation_function_d(x, a=1, b=1, type="sigmoid"):
    if type == "sigmoid":
        return sigmoid_d(x, a, b)
    elif type == "softmax":
        return softmax_d(x)
    elif type == "relu":
        return relu_d(x)
    elif type == "tanh":
        return tanh_d(x)


def loss_function(targets, predictions, type="cross entropy"):
    if type == "mse":
        return mse(targets, predictions)
    elif type == "cross entropy":
        return cross_entropy(targets, predictions)


def loss_function_d(targets, predictions, type="cross entropy"):
    if type == "mse":
        return mse_d(targets, predictions)
    elif type == "cross entropy":
        return cross_entropy_d(targets, predictions)


def sigmoid(x, a, b):
    res = b / (1 + np.exp(-a * x))
    return res


def sigmoid_d(y, a, b):
    res = a * y * (1 - y / b)
    return res


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    sum_exp_x = np.sum(exp_x)
    res = exp_x / sum_exp_x
    return res


def softmax_d(x):
    return


def relu(x):
    res = np.maximum(0, x)
    return res


def relu_d(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x



def tanh(x):
    res = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return res


def tanh_d(x):
    res = 1 - x**2
    return res


def mse(targets, predictions):
    res = np.mean(1 / 2 * (predictions - targets) ** 2)
    return res


def mse_d(targets, predictions):
    res = predictions - targets
    return res


def cross_entropy(targets, predictions, epsilon=1e-12):
    # predictions = np.clip(predictions, epsilon, 1. - epsilon)
    # N = predictions.shape[0]
    # ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    res = -np.sum(targets * np.log(predictions + 1e-9))
    return res


def cross_entropy_d(targets, predictions):
    res = -targets / predictions
    return res






