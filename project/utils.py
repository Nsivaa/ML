import numpy as np
import pandas as pd
from multiprocessing import Process, Manager, Lock, cpu_count
from NeuralNetwork import *
from matplotlib import pyplot as plt
import itertools

import TenMaxPriorityQueue as minQueue


def get_search_space(grid):
    tuple_search_space = list(itertools.product(grid["eta"], grid["mb"], grid["momentum"], grid["n_layers"], grid["n_neurons"], grid["epochs"],
                                                grid["clip_value"], grid["hid_act_fun"], grid["out_act_fun"], grid[
                                                    "cost_fun"], grid["ridge_lambda"], grid["lasso_lambda"], grid["decay_max_steps"], grid["decay_min_value"]))
    dict_search_space = []
    for conf in tuple_search_space:
        dict_search_space.append(dict(zip(grid.keys(), conf)))

    return dict_search_space


def parallel_grid_search(k, data, search_space, n_inputs, n_outputs,refined=False,epochs_refinment=1000):
    cpus = cpu_count()
    if len(search_space) >= cpus:
        n_cores = cpus
    else:
        n_cores = len(search_space)

    split_search_space = np.array_split(search_space, n_cores)
    processes = []
    manager = Manager()
    lock = Lock()
    res = manager.list([[(10000,(100,{}))]])
    for i in np.arange(n_cores):
        processes.append(Process(target=grid_search,
                                 args=(k, data, split_search_space[i], n_inputs,
                                       n_outputs, res, lock)))
        processes[i].start()

    for process in processes:
        process.join()

    print("GRID SEARCH FINISHED")
    if refined:
        print("Results' refinment...")
        search_space=[]
        for elem in res[0]:
            (err,(variance,parameters)) = elem
            parameters["epochs"]=epochs_refinment
            search_space.append(parameters)
        parallel_grid_search(10,data,search_space,n_inputs,n_outputs)

    else:
        minQueue.printQueue(res[0])
        f = open("./results.txt", "w")
        minQueue.printQueue(res[0],file=f)
        f.close()



def grid_search(k, data, search_space, n_inputs, n_outputs, shared_res=None, lock=None):
    best_comb = []
    for parameters in search_space:
        n_layers = parameters["n_layers"]
        n_neurons = parameters["n_neurons"]
        net = NeuralNetwork()
        net.add_input_layer(n_inputs)
        net.add_hidden_layer(n_inputs, n_neurons)
        for _ in np.arange(n_layers - 1):
            net.add_hidden_layer(n_neurons, n_neurons)

        net.add_output_layer(n_neurons, n_outputs)
        err, variance = net.k_fold(k, data, parameters)
        minQueue.push(best_comb,(err,(variance,parameters)))

    if shared_res is not None:
        lock.acquire()
        temp=shared_res[0]
        for i in range(len(best_comb)):
            minQueue.push(temp,best_comb[i])

        shared_res[0]=temp
        lock.release()
        return best_comb

    else:
        print("GRID SEARCH FINISHED")
        minQueue.printQueue(best_comb)
        return 0

def compare_models(n, data, parameters, n_inputs, n_outputs):
    nets_errors = []
    min_err = 10**5
    n_layers = parameters["n_layers"]
    n_neurons = parameters["n_neurons"]
    for _ in np.arange(n):

        net = NeuralNetwork()
        net.add_input_layer(n_inputs, randomize_weights=True)
        net.add_hidden_layer(n_inputs, n_neurons, randomize_weights=True)
        for _ in np.arange(n_layers - 1):
            net.add_hidden_layer(n_neurons, n_neurons, randomize_weights=True)
        print(f"HIDDEN LAYER:{net.hidden_layers[0]}")
        net.add_output_layer(n_neurons, n_outputs, randomize_weights=True)
        initial_net = net #SAVE THE NON-TRAINED NET
        err = net.hold_out(data, parameters, randomize_shuffle=False)
        nets_errors.append(err)
        if err < min_err:
            min_err = err
            best_net = initial_net

    variance = np.var(nets_errors)
    bias = np.mean(nets_errors)
    return best_net, variance, bias


def plot_loss(losses: np.ndarray, cost_fun: str,test_losses=None):
    iterations = np.arange(len(losses))
    plt.title("Learning Curve")
    plt.xlabel("Epochs")
    if cost_fun == "mse":
        plt.ylabel("MSE Loss")
    elif cost_fun == "b_ce":
        plt.ylabel("Cross Entropy Loss")

    plt.plot(iterations, losses, color="green")
    if test_losses is not None:
        iterations = np.arange(len(test_losses))
        plt.plot(iterations, test_losses, color="black")
    plt.legend()
    plt.show
    
def plot_loss_Monk(losses: np.ndarray, cost_fun: str,ax,test_losses=None):
    iterations = np.arange(len(losses))
    ax.set_xlabel("Epochs")
    if cost_fun == "mse":
        ax.set_ylabel("MSE Loss")
        ax.set_title("Learning Curve")
        label1="training loss"
        label2="test loss"
    elif cost_fun == "acc":
        ax.set_title("Learning Curve")
        ax.set_ylabel("Accuracy")
        label1="training accuracy"
        label2="test accuracy"

    loss= ax.plot(iterations, losses, color="black", label = label1)
    if test_losses is not None:
        iterations = np.arange(len(test_losses))
        test=ax.plot(iterations, test_losses, color="red",linestyle='--',label= label2)
    ax.legend()


def process_monk_data(data: pd.DataFrame):
    np.random.seed(0)
    data = data.sample(frac=1)
    data.reset_index(drop=True, inplace=True)
    data = pd.get_dummies(
        data, columns=["a1", "a2", "a3", "a4", "a5", "a6"], dtype=int)
    return data


