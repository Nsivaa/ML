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
            self.weights = self.weights[1]
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

    def feed_forward(self, inputs: np.ndarray):
        output = np.dot(inputs, self.weights) + self.biases
        self.output = np.maximum(0, output)

    def __len__(self): #returns the number of nodes
        return len(self.biases[0])
    
    def __str__(self):
        out_str = ""
        for  pos, neur in enumerate(zip(self.weights.T, self.biases[0])):
            out_str += f"NODE {pos} WEIGHTS = "
            if type(neur[0]) == np.float64: #If weights is a 1D array (which happens in the first layer), its .T (transpose) returns a float and not an array
                for w in self.weights:      # So we iterate on the non-transposed version  
                    out_str += f"{w}, "
            else:
                for w in neur[0]:
                    out_str += f"{w}, "
            out_str += f" BIAS = {neur[1]}\n"
        return out_str
        
class NeuralNetwork:
    '''
        NeuralNetwork class contains:
        input_layer : Layer object
        hidden_layers: list of Layer objects
        output_layer: Layer object
    '''
    def __init__(self, input_layer: Layer = None, hidden_layers: list = None, output_layer : Layer = None):
        
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers

        if hidden_layers is None:
            self.hidden_layers = []
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

    def train(self, data: pd.DataFrame):
        for row in data.itertuples(index = False, name = None): 
            self.input_layer.weights = np.asarray(row) #FIRST HIDDEN LAYER TAKES WEIGHTS FROM INPUT LAYER

            self.hidden_layers[0].feed_forward(self.input_layer.weights)

            for pos, layer in enumerate(self.hidden_layers[1:]):

                layer.feed_forward(self.hidden_layers[pos].output)
            
            self.output_layer.feed_forward(self.hidden_layers[-1].output)

        return

    '''Return the number of layers of the network'''
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
        res += len(self.input_layer)
        for layer in self.hidden_layers:
            res += len(layer)
        res += len(self.output_layer)

        return res

    ''' Print the nodes '''
    
    def __str__(self):
        res = f"INPUT LAYER: \n{str(self.input_layer)}\n"

        for pos, layer in enumerate(self.hidden_layers):
            res += f"LAYER {pos} \n" + str(layer) + "\n"
        res += "\n"
        res += f"OUTPUT LAYER: \n{str(self.output_layer)} "

        return res
    
    
