import numpy as np
import pandas as pd
from nn_utils import *
from matplotlib import pyplot as plt

def relu(x):
    return np.where(x > 0, x, 0.01 * x)


def D_relu(x):
    return np.where(x > 0, 1, 0)


def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)


def D_leaky_relu(x):
    return np.where(x > 0, 1, 0.01)


def tanh(x):
    return np.tanh(x)


def D_tanh(x):
    return 1. - np.tanh(x) ** 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def D_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def activation(arr: np.ndarray, act_fun: str):
    if act_fun == "relu":
        return relu(arr)
    elif act_fun == "leaky_relu":
        return leaky_relu(arr)
    elif act_fun == "sigmoid":
        return sigmoid(arr)
    elif act_fun == "tanh":
        return tanh(arr)
    elif act_fun == "linear":
        return arr
    else:
        print("Invalid activation function")
        return None


def derivative(act_fun: str, arr: np.ndarray):
    if act_fun == "relu":
        return D_relu(arr)
    elif act_fun == "leaky_relu":
        return D_leaky_relu(arr)
    elif act_fun == "sigmoid":
        return D_sigmoid(arr)
    elif act_fun == "tanh":
        return D_tanh(arr)
    elif act_fun == "linear":
        return arr
    else:
        print("Invalid activation function")
        return None


def bin_cross_entropy(netOut: float, sampleOut: int):
    '''
     netOut is the predicted probability that the item is of positive class (class 1) 

     When the observation belongs to class 1 (sampleOut = 1) the first part of the formula becomes active
     and the second part vanishes and vice versa in the case observation's actual class is 0.
    '''
    return -(sampleOut * np.log(netOut) + (1 - sampleOut) * np.log(1 - netOut))


def mse(netOut: np.ndarray, sampleOut: np.ndarray):
    s = 0
    for i in np.arange(0, netOut.size):
        s += np.square(netOut[i] - sampleOut[i])
    return 0.5 * s


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

def process_monk_data(tr_data: pd.DataFrame, test_data:pd.DataFrame = None):
    tr_data = tr_data.sample(frac=1)
    tr_data.reset_index(drop=True, inplace=True)
    tr_data_Y = tr_data[["Class"]]
    tr_data_X = one_hot_encode(tr_data)

    if test_data is not None:        
        test_data = test_data.sample(frac=1)
        test_data.reset_index(drop = True, inplace = True)
        test_data_Y = test_data[["Class"]]
        test_data_X = one_hot_encode(test_data)
        
        common_cols = list(set(tr_data.columns).intersection(test_data.columns))
        set_test_unknowns(test_data_X, common_cols)

        return tr_data_X, tr_data_Y, test_data_X, test_data_Y
    
    return tr_data_X, tr_data_Y


def one_hot_encode(data: pd.DataFrame):
    tr_id_dummies = pd.get_dummies(data["ID"], dtype=int, drop_first = True)
    tr_data_X = pd.concat([data.drop(["Class", "ID"], axis = 1), tr_id_dummies], axis = 1)
    tr_data_X.insert(tr_data_X.shape[0], "unknown", 0)
    return tr_data_X
    

def set_test_unknowns(test_data_X: pd.DataFrame, common_cols):
    for col in test_data_X.columns: 
        if col not in common_cols: #IF IT'S AN UNSEEN CLASS WE SET "UNKNOWN" TO 1 AND REMOVE THE COLUMN
            row = test_data_X.loc[test_data_X[col] == 1].index[0] 
            test_data_X.at[row,"unknown"] = 1
            test_data_X.drop(col, axis = 1, inplace = True)
    print(test_data_X)