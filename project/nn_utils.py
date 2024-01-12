import numpy as np
import pandas as pd
import matplotlib as plt

'''Returns the max among x and 0'''
def relu(x):
    return x * (x > 0)    

'''If the value is greater than 0, we leave it as is, else we replace it with value * 0,01'''
def leaky_relu(x):
    if x > 0:
        return x
    return x * 0.01 

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Layer:
    '''
        The Layer class represents a layer of the neural network.
        It has a list of Nodes
    '''
    def __init__(self, weights: np.ndarray = None, biases: np.ndarray = None):
        
        if weights is not None and biases is not None:
            if weights.shape[0] != biases.shape[0]:
                print("ERROR: biases and weights have different dimensions")
                return
        self.weights = weights
        self.biases = biases
        
    def add_neuron(self, neuron_weights: np.ndarray = None, neuron_bias: int = None):
        if neuron_weights is not None:
            if len(neuron_weights.shape) > 1:
                print("ERROR: Trying to append more than a neuron")
                return
        
        np.append(self.weights, neuron_weights) 
        np.append(self.biases, neuron_bias)
        
    def __len__(self):
        return len(self.biases)

    def __str__(self):
        out_str = ""
        for pos, weights in enumerate(el for el in self.weights if el is not None):
            out_str += f"node {pos} weights = " + str(weights) + f", bias = {self.biases[pos]}\n"
        return out_str

class NeuralNetwork:
    '''
        NeuralNetwork class contains a list of Layers and definition of all the parameters
    '''
    def __init__(self, layers = None):
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

    ''' Add a new layer in the network. If the position is not specified, it is appended'''
    def add_layer(self, layer, pos = None):
        if pos is None:
            self.layers.append(layer)
        else:
            self.layers.insert(pos, layer)

    '''Return the number of nodes of the network'''
    def __len__(self):
        res = 0
        for layer in self.layers:
            res += len(layer)
        return res

    ''' Print the nodes '''
    def __str__(self):
        res = ""
        for pos, layer in enumerate(self.layers):
            res += f"LAYER {pos} \n" + str(layer) + "\n"
        return res

def feed_forward(A_prev : np.ndarray, W : np.ndarray, b : np.ndarray):
        A = np.dot(W, A_prev) + b
        return A
    
