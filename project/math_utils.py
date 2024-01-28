import numpy as np

def relu(x):
    return np.where(x > 0, x, 0)


def D_relu(x):
    return np.where(x > 0, 1, 0)


def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)


def D_leaky_relu(x):
    return np.where(x > 0, 1, 0.01)


def tanh(x):
    return np.tanh(x)


def D_tanh(x):
    return 1 - np.tanh(x) ** 2


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

#retituisce 1 per match, 0 else
def accuracy(netOut, sampleOut):
    if netOut >= 0.5:
        netOut =1
    else:
        netOut = 0

    if netOut == sampleOut:
        return 1
    return 0
