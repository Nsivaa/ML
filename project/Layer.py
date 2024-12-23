import numpy as np
from math_utils import activation


class Layer:
    '''
        The Layer class represents a layer of the neural network.
        It has a ndarray of weights and an ndarray of biases, and a boolean "is_input" attribute
        n_input rappresents the number of neurons of the precedent layer which entails the number of weights of neuron in the layer
        The input layer is modelled as a layer with 1 input and n neurons so its weights are a single row and are actually the input values
    '''

    def __init__(self, n_inputs: int = 0, n_neurons: int = 0, is_input: bool = False, randomize_weights: bool = False,
                 type="monk"):
        self.is_input = is_input
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.acc_bias_gradients = None
        self.acc_weight_gradients = None
        self.bias_gradients = None
        self.weight_gradients = None
        self.output = np.zeros(n_neurons)
        self.type = type

        if self.is_input:
            self.weights = np.zeros(n_neurons)
        else:
            self.weights = np.zeros((n_inputs, n_neurons))
            if not randomize_weights:
                np.random.seed(0)
            else:
                np.random.seed()
            # xavier normalization for weight initialization
            shape = self.weights.shape
            if self.type == "monk":
                self.weights = np.random.randn(
                    *shape) * np.sqrt(2.0 / (shape[0] + shape[1]))
            else:
                # he initialization
                self.weights = np.random.randn(
                    *shape) * np.sqrt(2.0 / shape[0])

            self.momentum_velocity_w = np.zeros(
                (self.weights.shape[0], self.weights.shape[1]))

            self.biases = np.zeros((1, n_neurons))
            self.momentum_velocity_b = np.zeros(n_neurons)

        return



    def feed_forward(self, inputs: np.ndarray, act_fun: str):
        output = np.dot(inputs, self.weights) + self.biases[0]
        self.output = activation(output, act_fun)

    def clip_gradients(self, clip_value):
        self.acc_weight_gradients = np.clip(
            self.acc_weight_gradients, -clip_value, clip_value)
        self.acc_bias_gradients = np.clip(
            self.bias_gradients, -clip_value, clip_value)

    def __len__(self):  # returns the number of nodes
        return len(self.biases[0])

    def __str__(self):
        out_str = ""
        if self.is_input:
            for i in np.arange(0, self.n_neurons):
                out_str += f"NODE {i} OUTPUT = {self.output[i]}\n"
        else:
            for pos, neur in enumerate(zip(self.weights.T, self.biases[0])):
                out_str += f"NODE {pos} WEIGHTS = "
                # If weights is a 1D array (which happens in the first layer), its .T (transpose) returns a float and not an array
                if type(neur[0]) == np.float64:
                    for w in self.weights:  # So we iterate on the non-transposed version
                        out_str += f"{w}, "
                else:
                    for w in neur[0]:
                        out_str += f"{w}, "
                out_str += f" BIAS = {neur[1]}\n"
        return out_str
