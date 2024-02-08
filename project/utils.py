import numpy as np
import pandas as pd
from multiprocessing import Process, Manager, Lock, cpu_count
from NeuralNetwork import *
import os
import random
import string
import itertools

import TenMaxPriorityQueue as minQueue


def get_search_space(grid):
    tuple_search_space = list(itertools.product(grid["eta"], grid["mb"], grid["momentum"], grid["n_layers"], grid["n_neurons"], grid["epochs"],
                                                grid["clip_value"], grid["hid_act_fun"], grid["out_act_fun"], grid[
                                                    "cost_fun"], grid["ridge_lambda"], grid["lasso_lambda"], grid["decay_max_steps"], grid["decay_min_value"], grid["es_patience"]))
    dict_search_space = []
    for conf in tuple_search_space:
        dict_search_space.append(dict(zip(grid.keys(), conf)))

    random.shuffle(dict_search_space)
    return dict_search_space


def parallel_grid_search(k, data, es_data, search_space, n_inputs, n_outputs, refined=False, epochs_refinment=1000, type="monk", verbose="no"):
    # %todo: controllare che abbai senso fare sta roba effettivamente
    cpus = int(cpu_count() / 2 - 1)
    if len(search_space) >= cpus:
        n_cores = cpus
    else:
        n_cores = len(search_space)
    print(f"N_cores = {n_cores}")
    dirName = None
    if verbose == "yes":
        dirName = "gridResults_" + \
            "".join(random.choice(string.ascii_letters) for _ in range(10))
        os.makedirs(dirName)
    split_search_space = np.array_split(search_space, n_cores)
    processes = []
    manager = Manager()
    lock = Lock()
    res = manager.list([[(-1000000, (100, 100, {}))]])
    for i in np.arange(n_cores):
        processes.append(Process(target=grid_search,
                                 args=(k, data, es_data, split_search_space[i], n_inputs,
                                       n_outputs, res, lock, type, verbose, dirName)))
        processes[i].start()

    for process in processes:
        process.join()

    print("GRID SEARCH FINISHED")
    if refined:
        print("Results' refinment...")
        search_space = []
        for elem in res[0]:
            (err, (variance, parameters)) = elem
            parameters["epochs"] = epochs_refinment
            search_space.append(parameters)
        parallel_grid_search(
            10, data, es_data, search_space, n_inputs, n_outputs)

    else:
        minQueue.printQueue(res[0])
        f = open("./results.txt", "w")
        minQueue.printQueue(res[0], file=f)
        f.close()


# Si suppone se sia eseguita solo su CUP, sempre con ES
def grid_search(k, data, es_data, search_space, n_inputs, n_outputs, shared_res=None, lock=None, type="monk", verbose="no", dirName=None):
    # minqueue usata per tenere traccia delle 10 migliori combinazioni
    best_comb = []
    for parameters in search_space:
        n_layers = parameters["n_layers"]
        n_neurons = parameters["n_neurons"]
        net = NeuralNetwork(type=type)
        net.add_input_layer(n_inputs)
        net.add_hidden_layer(n_inputs, n_neurons)
        for _ in np.arange(n_layers - 1):
            net.add_hidden_layer(n_neurons, n_neurons)

        net.add_output_layer(n_neurons, n_outputs)
        # tr_sarà l'errore scesi sotto il quale il modello rischia di overfittare quindi bisgona interrompere il training
        tr_err, valid_err, valid_variance = net.k_fold(
            k, data, parameters, es_data=es_data)
        minQueue.push(
            best_comb, (valid_err, (valid_variance, tr_err, parameters)))
        if verbose == "yes":
            randName = dirName+"/" + \
                "".join(random.choice(string.ascii_letters) for _ in range(15))
            file = open(randName+".txt", "w")
            print(f"{parameters}\nValidation mean = {valid_err}, Variance = {valid_variance}\nTraining mean (ES) = {tr_err}\n", file=file)
            file.close()

    if shared_res is not None:
        lock.acquire()
        temp = shared_res[0]
        for i in range(len(best_comb)):
            minQueue.push(temp, best_comb[i])

        shared_res[0] = temp
        lock.release()

    else:
        print("GRID SEARCH FINISHED")
        minQueue.printQueue(best_comb)

def weight_average(nets):
    # assumes every net in the list has the same number of neurons

    final_net = NeuralNetwork(type="cup")
    final_net.add_input_layer(nets[0].input_layer.n_neurons)
    for i in np.arange(len(nets[0].hidden_layers)):
        h_layer = Layer(nets[0].hidden_layers[i].n_inputs,
                        nets[0].hidden_layers[i].n_neurons, type="cup")
        h_layer.weights = np.mean(
            [net.hidden_layers[i].weights for net in nets], axis=0)
        h_layer.biases = np.mean(
            [net.hidden_layers[i].biases for net in nets], axis=1)
        final_net.hidden_layers.append(h_layer)

    final_net.add_output_layer(
        nets[0].input_layer.n_inputs, nets[0].output_layer.n_neurons)
    final_net.output_layer.weights = np.mean(
        [net.output_layer.weights for net in nets], axis=0)
    final_net.output_layer.biases = np.mean(
        [net.output_layer.biases for net in nets], axis=1)

    return final_net

def plot_loss_Cup(losses: np.ndarray, cost_fun: str, ax, test_losses=None):
    iterations = np.arange(len(losses))
    ax.set_xlabel("Epochs")
    if cost_fun == "mse":
        ax.set_ylabel("MSE Loss")
        ax.set_title("Learning Curve")
        label1 = "training loss"
        label2 = "test loss"
    else:
        ax.set_ylabel("MEE Error")
        ax.set_title("Learning Curve")
        label1 = "training error"
        label2 = "test error"

    loss = ax.plot(iterations, losses, color="black", label=label1)
    if test_losses is not None:
        iterations = np.arange(len(test_losses))
        test = ax.plot(iterations, test_losses, color="red",
                       linestyle='--', label=label2)
    ax.legend()


def plot_loss_Monk(losses: np.ndarray, cost_fun: str, ax, test_losses=None):
    iterations = np.arange(len(losses))
    ax.set_xlabel("Epochs")
    if cost_fun == "mse":
        ax.set_ylabel("MSE Loss")
        ax.set_title("Learning Curve")
        label1 = "training loss"
        label2 = "test loss"
    else:
        ax.set_ylabel("Accuracy")
        ax.set_title("Learning Curve")
        label1 = "training accuracy"
        label2 = "test accuracy"

    loss = ax.plot(iterations, losses, color="black", label=label1)
    if test_losses is not None:
        iterations = np.arange(len(test_losses))
        test = ax.plot(iterations, test_losses, color="red",
                       linestyle='--', label=label2)
    ax.legend()


def process_monk_data(data: pd.DataFrame):
    np.random.seed(0)
    data = data.sample(frac=1)
    data.reset_index(drop=True, inplace=True)
    data = pd.get_dummies(
        data, columns=["a1", "a2", "a3", "a4", "a5", "a6"], dtype=int)
    return data

