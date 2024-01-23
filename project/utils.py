import numpy as np
import pandas as pd
from multiprocessing import Process, Manager, Lock, cpu_count
from NeuralNetwork import *
from matplotlib import pyplot as plt
import itertools


def get_search_space(grid):
    tuple_search_space = list(itertools.product(grid["eta"], grid["mb"], grid["momentum"], grid["n_layers"], grid["n_neurons"], grid["epochs"],
                                                grid["clip_value"], grid["hid_act_fun"], grid["out_act_fun"], grid[
                                                    "cost_fun"], grid["ridge_lambda"], grid["lasso_lambda"],
                                                grid["linear_decay"], grid["eta0"], grid["max_steps"], grid["epochs_update"]))
    dict_search_space = []
    for conf in tuple_search_space:
        dict_search_space.append(dict(zip(grid.keys(), conf)))

    return dict_search_space


def parallel_grid_search(k, data, search_space, n_inputs, n_outputs):
    n_cores = cpu_count()
    split_search_space = np.array_split(search_space, n_cores)
    processes = []
    manager = Manager()
    lock = Lock()
    res = manager.list(["", 10**5])
    for i in np.arange(n_cores):
        processes.append(Process(target=grid_search,
                              args=(k, data, split_search_space[i], n_inputs,
                                    n_outputs, res, lock)))
        processes[i].start()

    for process in processes:
        process.join()

    print("GRID SEARCH FINISHED")

    f = open("./results.txt", "w")
    f.write(str(res))
    f.close()
    return res


def grid_search(k, data, search_space, n_inputs, n_outputs, shared_res=None, lock=None):
    np.random.seed(0)
    min_err = 10 ** 5
    val_errors = {}
    for parameters in search_space:
        n_layers = parameters["n_layers"]
        n_neurons = parameters["n_neurons"]
        net = NeuralNetwork()
        net.add_input_layer(n_inputs)
        net.add_hidden_layer(n_inputs, n_neurons)
        for _ in np.arange(n_layers - 1):
            net.add_hidden_layer(n_neurons, n_neurons)

        net.add_output_layer(n_neurons, n_outputs)
        err = net.k_fold(k, data, parameters)
        # frozenset because dict is not hashable
        #val_errors[frozenset(parameters.items())] = err

        if err < min_err:
            best_conf = parameters
            min_err = err

    if shared_res is not None:
        lock.acquire()
        if min_err < shared_res[1]:
            shared_res[0] = parameters
            shared_res[1] = min_err
        lock.release()
        return 

    else:
        print("GRID SEARCH FINISHED")
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
