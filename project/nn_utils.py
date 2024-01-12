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
        It has a ndarray of weights and an ndarray of biases
    '''
    def __init__(self, weights: np.ndarray = None, biases: np.ndarray = None, is_input = False):    
        if not is_input: #if the layer is an input layer we skip the dimension constraint check
            if weights is not None and biases is not None and weights.shape[1] != biases.shape[0]:
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
    def __init__(self, input_layer: Layer = None, hidden_layers: list = None):
        
        self.input_layer = input_layer
        if hidden_layers is not None:
            self.hidden_layers = hidden_layers
        else:
            self.hidden_layers = []
        return

    def add_input_layer(self, n_neurons : int = None):
        if self.input_layer is not None:
            print("Warning: input layer already present. Overwriting")
        weights = np.random.rand(n_neurons)
        biases = np.random.rand(n_neurons)
        input_layer = Layer(weights=weights, biases=biases, is_input=True)
        self.input_layer = input_layer
        return
    ''' 
        Add a new layer in the network. If the position is not specified, it is appended.
        The layer is created with the number of weights for each neuron relative to the previous layer
    '''
    def add_hidden_layer(self, n_neurons: int = None, pos: int = None):
        if pos is None:
            if self.hidden_layers is not None:
                pos = len(self.hidden_layers)

        if len(self.hidden_layers) == 0 or self.hidden_layers is None:
            weights = np.random.rand(len(self.input_layer), n_neurons)

        else:
            weights = np.random.rand(len(self.hidden_layers[pos - 1]), n_neurons)

        biases = np.random.rand(n_neurons)
        layer = Layer(weights, biases)
        self.hidden_layers.insert(pos, layer)
        return
        
    '''Return the number of nodes of the network'''
    def __len__(self):
        res = 0
        if self.input_layer is not None:
            res += 1

        if self.hidden_layers is not None:
            res += len(self.hidden_layers)
        
        #if self.output_layer is not None:
        #    res+= 1

        return res

    def number_of_nodes(self):
        res = 0
        if self.input_layer is not None:
            res += len(self.input_layer)
        
        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                res += len(layer)

        #if self.output_layer is not None:
        #    res+= len(self.output_layer)

        return res

    ''' Print the nodes '''
    def __str__(self):
        res = str(self.input_layer)
        for pos, layer in enumerate(self.hidden_layers):
            res += f"LAYER {pos} \n" + str(layer) + "\n"
        return res

def feed_forward(A_prev : np.ndarray, W : np.ndarray, b : np.ndarray):
        A = np.dot(W, A_prev) + b
        return A
    
