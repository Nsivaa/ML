import numpy as np
import pandas as pd
from multiprocessing import Process, Manager, Lock, cpu_count
from NeuralNetwork import *
import os
import random
import itertools
import datetime
from tqdm import tqdm
from Ensemble import *
import matplotlib.pyplot as plt
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


def parallel_grid_search(k, data,es_data, search_space, n_inputs, n_outputs, type="monk", verbose="yes"):
    cpus = (os.cpu_count()) -2
    if len(search_space) >= cpus:
        n_cores = cpus
    else:
        n_cores = len(search_space)
    print(f"N_cores = {n_cores}")
    dirName = None
    if verbose == "yes":
        current_time = datetime.datetime.now()
        time = f"{current_time.month}_{current_time.day}_{current_time.hour }_{current_time.minute}_{current_time.second}"
        dirName="gridResults/" + time
        os.makedirs(dirName)
    split_search_space = np.array_split(search_space, n_cores)
    processes = []
    manager = Manager()
    lock = Lock()
    res = manager.list([[(-np.inf, (100, 100, {}))]])
    for i in np.arange(n_cores):
        processes.append(Process(target=grid_search,
                                 args=(k, data, es_data, split_search_space[i], n_inputs,
                                       n_outputs, res, lock, type, verbose, dirName)))
        processes[i].start()

    for process in processes:
        process.join()

    print("GRID SEARCH FINISHED")
    minQueue.printQueue(res[0])


# always executed with es_data
def grid_search(k, data,es_data, search_space, n_inputs, n_outputs, shared_res=None, lock=None, type="monk", verbose="no", dirName=None):
    # minqueue is used to store the best 10 combinations
    best_comb = []
    for parameters in search_space:
        tr_err,valid_err, valid_variance = k_fold(
            k, data, parameters,es_data,type,n_inputs,n_outputs)
        minQueue.push(
            best_comb,(valid_err,(valid_variance,tr_err,parameters)))
        
        if verbose == "yes":
            fileName = dirName + "/" + ",".join(str(v) for v in parameters.values())
            file = open(fileName + ".txt", "w")
            print(f"{parameters}\nValidation mean = {valid_err}, Variance = {valid_variance}\nTraining mean (ES) = {tr_err}\n", file=file)
            file.close()


    if shared_res is not None:
        lock.acquire()
        temp = shared_res[0]
        for i in range(len(best_comb)):
            val_mean, (variance,tr_mean,comb) = best_comb[i]
            val_mean = val_mean*(-1)
            minQueue.push(temp,(val_mean, (variance,tr_mean,comb)))

        shared_res[0]=temp
        lock.release()

    else:
        print("GRID SEARCH FINISHED")
        minQueue.printQueue(best_comb)

def k_fold(k, data, parameters,es_data,type,n_inputs,n_outputs,es_stop=None, progress_bar=True):

    if "ID" in data.columns:
        data.drop(["ID"], axis=1, inplace=True)
    np.random.seed()
    data = data.sample(frac=1)
    folds = np.array_split(data, k)
    valid_errors = []
    tr_errors = []
    for fold in tqdm(folds, desc="Folds", disable=not progress_bar):
        n_layers = parameters["n_layers"]
        n_neurons = parameters["n_neurons"]
        net = NeuralNetwork(type=type)
        net.add_input_layer(n_inputs)
        net.add_hidden_layer(n_inputs, n_neurons)
        for _ in np.arange(n_layers - 1):
            net.add_hidden_layer(n_neurons, n_neurons)

        net.add_output_layer(n_neurons, n_outputs)
        tr_set = pd.concat(
            [f for f in folds if not (pd.Series.equals(f, fold))], axis=0)
        valid_error, tr_error= net.train(tr_set, parameters,test_data=fold,es_data=es_data, progress_bar=False,es_stop=es_stop)
        valid_errors.append(valid_error)
        tr_errors.append(tr_error)

    valid_errors = np.array(valid_errors)
    tr_errors = np.array(tr_errors)
    valid_mean = valid_errors.mean()
    tr_mean = tr_errors.mean()
    valid_var = valid_errors.var()
    print(f"val_mean = {valid_mean}")

    return tr_mean,valid_mean, valid_var

# the difference between k_fold method is only the inizializzation of Ensemble object instead of NeuralNetwork
def k_fold_ensemble(k, data,structures, train_params,progress_bar=True,epochs=2000):

    if "ID" in data.columns:
        data.drop(["ID"], axis=1, inplace=True)
    np.random.seed()
    data = data.sample(frac=1)
    folds = np.array_split(data, k)
    valid_errors = []
    tr_errors = []
    for fold in tqdm(folds, desc="Folds", disable=not progress_bar):
        ensemble= Ensemble(structures)
        tr_set = pd.concat(
            [f for f in folds if not (pd.Series.equals(f, fold))], axis=0)
        valid_error, tr_error= ensemble.train_models(tr_set, train_params,test_data=fold,epochs=epochs)
        valid_errors.append(valid_error)
        tr_errors.append(tr_error)

    valid_errors = np.array(valid_errors)
    tr_errors = np.array(tr_errors)
    valid_mean = valid_errors.mean()
    tr_mean = tr_errors.mean()
    valid_var = valid_errors.var()
    print(f"val_mean = {valid_mean}")

    return tr_mean,valid_mean, valid_var

def plot_loss_Cup(losses: np.ndarray, cost_fun: str,ax,test_losses=None):
    iterations = np.arange(len(losses))
    ax.set_xlabel("Epochs")
    if cost_fun == "mse":
        ax.set_ylabel("MSE Loss")
        ax.set_title("Learning Curve")
        label1 ="training loss"
        label2 ="test loss"
    else:
        ax.set_ylabel("MEE Error")
        ax.set_title("Learning Curve")
        label1 = "training error"
        label2 = "test error"

    loss = ax.plot(iterations, losses, color="black", label = label1)
    if test_losses is not None:
        iterations = np.arange(len(test_losses))
        test = ax.plot(iterations, test_losses, color="red",linestyle='--',label= label2)
    ax.legend()

def plot_loss_Monk(losses: np.ndarray, cost_fun: str,ax,test_losses=None):
    iterations = np.arange(len(losses))
    ax.set_xlabel("Epochs")
    if cost_fun == "mse":
        ax.set_ylabel("MSE Loss")
        ax.set_title("Learning Curve")
        label1 ="training loss"
        label2 ="test loss"
    else:
        ax.set_ylabel("Accuracy")
        ax.set_title("Learning Curve")
        label1 = "training accuracy"
        label2 = "test accuracy"

    loss = ax.plot(iterations, losses, color="black", label = label1)
    if test_losses is not None:
        iterations = np.arange(len(test_losses))
        test = ax.plot(iterations, test_losses, color="red",linestyle='--',label= label2)
    ax.legend()


def plot_ensembles(test_mse,train_mse,test_mee,train_mee,n=5):
    fig,axs = plt.subplots(1,2,figsize=(10,5))
    #First we need to calculate the means
    train_mse_mean=get_mean(train_mse)
    train_mee_mean=get_mean(train_mee)
    test_mse_mean=get_mean(test_mse)
    test_mee_mean=get_mean(test_mee)
    iterations = np.arange(len(test_mse_mean))

    # MSE plot
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("MSE Loss")
    axs[0].set_title("Learning Curve")
    label1 ="ensemble models mean training MSE"
    label2 ="ensemble models mean test MSE"
    axs[0].plot(iterations, train_mse_mean, color="black", label = label1)
    axs[0].plot(iterations, test_mse_mean, color="red", linestyle="--", label = label2)
    axs[0].legend()

    # MEE plot
    label1 ="ensemble models mean training MEE"
    label2 ="ensemble models mean test MEE"
    axs[1].set_title("Learning Curve")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("MEE Loss")
    axs[1].set_ylim((-1,11))
    axs[1].plot(iterations, train_mee_mean, color="black", linestyle='-',label = label1)
    axs[1].plot(iterations, test_mee_mean, color="red", linestyle='--',label = label2)

    axs[1].legend()


    
    fig.tight_layout(pad=2.0)
    plt.show()
    return train_mse_mean, train_mee_mean, test_mse_mean,test_mee_mean

def process_monk_data(data: pd.DataFrame):
    np.random.seed(0)
    data = data.sample(frac=1)
    data.reset_index(drop=True, inplace=True)
    data = pd.get_dummies(
        data, columns=["a1", "a2", "a3", "a4", "a5", "a6"], dtype=int)
    return data

def get_mean(list):
    max_lenght = max(len(lista) for lista in list)
    # padding of the shorter lists within list
    _list = [lista + [0] * (max_lenght - len(lista)) for lista in list]
    mean = [sum(valori) / (sum(1 for val in valori if val > 0)) for valori in zip(*_list)]
    return mean

