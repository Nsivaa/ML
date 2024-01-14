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

np.random.seed(0)

class Layer:
    '''
        The Layer class represents a layer of the neural network.
        It has a ndarray of weights and an ndarray of biases, and a boolean "is_input" attribute 
    '''
    def __init__(self, n_inputs: int = 0, n_neurons: int = 0, is_input : bool = False):    
        self.is_input = is_input

        if n_neurons == 0:
            self.weights = np.empty((0,0))

        else:
            self.weights = 0,1 * np.random.randn(n_inputs, n_neurons)

        self.biases = np.zeros((1, n_neurons))
        return
    
    ''' 
    def add_neuron(self, neuron_weights: np.ndarray = None, neuron_bias: int = None):
        if neuron_weights is not None:
            if len(neuron_weights.shape) > 1:
                print("ERROR: Trying to append more than a neuron")
                return
        
        np.append(self.weights, neuron_weights) 
        np.append(self.biases, neuron_bias)
    '''

    def feed_forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def __len__(self): #returns the number of nodes
     
        return len(self.biases)

    def __str__(self):

        out_str = ""

        for pos, weights in enumerate(el for el in self.weights[1] if el is not None):
            out_str += f"NODE {pos} WEIGHTS = {str(weights[pos])}, BIAS = {self.biases[0][pos]}\n"

        return out_str

class NeuralNetwork:
    '''
        NeuralNetwork class contains:
        input_layer : Layer object
        hidden_layers: list of Layer objects
        output_layer: Layer object
    '''
    def __init__(self, input_layer: Layer = None, hidden_layers: list = [], output_layer : Layer = None):
        
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

        return

    def add_input_layer(self, n_neurons : int = 0):
        self.input_layer = Layer(n_inputs = 1, n_neurons = n_neurons, is_input = True)
        return
    
    ''' 
        Add a new layer in the network. If the position is not specified, it is appended.
        The layer is created with the number of weights for each neuron relative to the previous layer
    '''
    def add_hidden_layer(self, n_inputs: int = 0, n_neurons: int = 0, pos: int = -1):
        layer = Layer(n_inputs, n_neurons, is_input = False)
        if pos == -1:
            self.hidden_layers.append(layer)

        else:
            self.hidden_layers.insert(pos, layer)
        return
        
    def add_output_layer(self, n_inputs : int = 0, n_neurons : int = 0):
        self.output_layer = Layer(n_inputs, n_neurons, is_input = False)
        return 
    
    def neur_feed_forward(self, A_prev : np.ndarray, W : np.ndarray, b : np.ndarray):
            A = np.dot(W, A_prev) + b
            return A

    def train(self, data: pd.DataFrame):
        for row in data.itertuples(index = False, name = None): #iterrows instead?? 
            self.input_layer.weights = np.asarray(row) 
            
            #FIRST HIDDEN LAYER TAKES WEIGHTS FROM INPUT LAYER #
            for neur in zip(np.nditer(self.hidden_layers[0].weights), np.nditer(self.hidden_layers[0].biases)):
                    self.hidden_layers[0].weights = self.neur_feed_forward(self.input_layer.weights.shape[0], neur[0].T, neur[1])

            '''
            for pos, layer in enumerate(self.hidden_layers[1:]):

                for neur in zip(np.nditer(layer.weights), np.nditer(layer.biases)):
                    neur[0] = self.neur_feed_forward(self.input_layer.weights.shape[0], neur[0], neur[1])
            '''
        return

    '''Return the number of nodes of the network'''
    def __len__(self):
        res = 0
        if self.input_layer is not None:
            res += 1

        if self.hidden_layers is not None:
            res += len(self.hidden_layers)
        
        if self.output_layer is not None:
            res+= 1

        return res

    def number_of_nodes(self):
        res = 0
        if self.input_layer is not None:
            res += len(self.input_layer)
        
        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                res += len(layer)

        if self.output_layer is not None:
            res+= len(self.output_layer)

        return res

    ''' Print the nodes '''
    
    def __str__(self):
        res = f"INPUT LAYER: \n {str(self.input_layer)}\n "

        for pos, layer in enumerate(self.hidden_layers):
            res += f"LAYER {pos} \n" + str(layer) + "\n"
        res += "\n"
       # res += f"OUTPUT LAYER: \n {str(self.output_layer)} "

        return res
    
    
