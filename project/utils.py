import numpy as np
import pandas as pd

from NeuralNetwork import *
from Layer import Layer
from matplotlib import pyplot as plt


def grid_search(k, data, search_space, n_inputs, n_outputs):
    '''
    "eta" : 1,
    "mb" : 2,
    "momentum" : 3,
    "n_layers" : 4,
    "n_neurons" : 5
    "epochs" : 6,
    "hid_act_fun" : 7,
    "out_act_fun" : 8,
    "clip_value" : 9
    '''

    np.random.seed(0)
    val_errors = {}

    for parameters in search_space:
        hidden_layers = []
        input_layer = Layer(n_inputs)
        hidden_layers.append(Layer(n_inputs, parameters[5]))
        for _ in np.arange(parameters[4] - 1):
            hidden_layers.append(Layer(parameters[5], parameters[5]))

        output_layer = Layer(parameters[5], n_outputs)
        net = NeuralNetwork(input_layer, hidden_layers, output_layer)

        val_errors[parameters] = net.k_fold(k, data, parameters)

    # min_err = np.min(val_errors.values())
    # return val_errors[min_err]
    return val_errors


def plot_loss(losses: np.ndarray, cost_fun: str):
    iterations = np.arange(len(losses))
    plt.title("Learning Curve")
    plt.xlabel("Epochs")
    if cost_fun == "mse":
        plt.ylabel("MSE Loss")
    elif cost_fun == "b_ce":
        plt.ylabel("Cross Entropy Loss")

    plt.plot(iterations, losses, color="green")
    plt.show()


def process_monk_data(data: pd.DataFrame):
    data = data.sample(frac=1)
    data.reset_index(drop=True, inplace=True)
    data = pd.get_dummies(
        data, columns=["a1", "a2", "a3", "a4", "a5", "a6"], dtype=int)
    return data
