import numpy as np
import pandas as pd

from NeuralNetwork import *
from Layer import Layer
from matplotlib import pyplot as plt


def grid_search(k, data, search_space, n_inputs, n_outputs):

    np.random.seed(0)
    val_errors = {}
    min_err = 10 ** 5
    for i, parameters in enumerate(search_space):
        n_layers = parameters["n_layers"]
        n_neurons = parameters["n_neurons"]
        net = NeuralNetwork()
        net.add_input_layer(n_inputs)
        net.add_hidden_layer(n_inputs, n_neurons)
        for _ in np.arange(n_layers):
            net.add_hidden_layer(n_neurons, n_neurons)

        net.add_output_layer(n_neurons, n_outputs)
        print(f"PARAMETER CONFIG N.{i}")
        err = net.k_fold(k, data, parameters)
        val_errors[frozenset(parameters.items())] = err #frozenset because dict is not hashable
        
        if err < min_err:
            best_conf = parameters
            min_err = err

    return best_conf, min_err, val_errors


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
