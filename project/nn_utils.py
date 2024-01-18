import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# TODO: cambiare bias in bias[0](in modo da avere un vettore e non una matrice con una riga) e modificare tutto di conseguenza


# TODO: CAMBIARE ACTIVATION FUNCTION OUTPUT LAYER CLASSIFICAZIONE

def relu(x):
    return np.where(x > 0, x, 0.01 * x)


def D_relu(x):
    return np.where(x > 0, 1, 0.01 * 1)


def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)


def D_leaky_relu(x):
    return np.where(x > 0, 1, 0.01 * 1)


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


# np.random.seed(0)


class Layer:
    '''
        The Layer class represents a layer of the neural network.
        It has a ndarray of weights and an ndarray of biases, and a boolean "is_input" attribute
        n_input rappresents the number of neurons of the precedent layer which entails the number of weights of neuron in the layer
        The input layer is modelled as a layer with 1 input and n neurons so its weights are a single row and are actually the input values
    '''

    def __init__(self, n_inputs: int = 0, n_neurons: int = 0, is_input: bool = False):
        self.is_input = is_input
        self.n_input = n_inputs
        self.n_neurons = n_neurons
        self.acc_bias_gradients = None
        self.acc_weight_gradients = None
        self.bias_gradients = None
        self.weight_gradients = None
        self.output = np.zeros(n_neurons)


        if self.is_input:
            self.weights = np.zeros(n_neurons)
        else:
            self.weights = 0, 1 * np.random.randn(n_inputs, n_neurons)
            self.weights = self.weights[1]
            # xavier normalization for weight inizialization
            shape = self.weights.shape
            self.weights = np.random.randn(
                *shape) * np.sqrt(2.0 / (shape[0] + shape[1]))

            self.momentum_velocity_w = np.zeros((self.weights.shape[0],self.weights.shape[1]))

            self.biases = np.zeros((1, n_neurons))
            self.momentum_velocity_b = np.zeros(n_neurons)

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

    # output will be a numpy ndarray with dimension (n)
    # ndarray of shape (1,n) is different from ndarray of shape (n)
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


class NeuralNetwork:
    '''
        NeuralNetwork class contains:
        input_layer : Layer object
        hidden_layers: list of Layer objects
        output_layer: Layer object
    '''

    def __init__(self, input_layer: Layer = None, hidden_layers: list = None, output_layer: Layer = None):

        self.input_layer = input_layer
        self.hidden_layers = hidden_layers

        if hidden_layers is None:
            self.hidden_layers = []
        self.output_layer = output_layer

        return

    def add_input_layer(self, n_neurons: int = 0):
        self.input_layer = Layer(
            n_inputs=1, n_neurons=n_neurons, is_input=True)
        return

    ''' 
        Add a new layer in the network. If the position is not specified, it is appended.
        The layer is created with the number of weights for each neuron relative to the previous layer
    '''

    def add_hidden_layer(self, n_inputs: int = 0, n_neurons: int = 0, pos: int = -1):
        layer = Layer(n_inputs, n_neurons, is_input=False)
        if pos == -1:
            self.hidden_layers.append(layer)

        else:
            self.hidden_layers.insert(pos, layer)
        return

    def add_output_layer(self, n_inputs: int = 0, n_neurons: int = 0):
        self.output_layer = Layer(n_inputs, n_neurons, is_input=False)

        return

    # calcola il MSE sul dataset con i pesi e i bias correnti
    def calcError(self, data: pd.DataFrame, labels: pd.DataFrame,
                  hid_act_fun: str = None, out_act_fun: str = None, cost_fun: str = None):
        totErr = 0
        for row, label in zip(data.itertuples(index=False, name=None), labels.itertuples(index=False, name=None)):
            totErr += self.forwardPropagation(row, label, hid_act_fun,
                                              out_act_fun, cost_fun)
        totErr /= data.shape[0]

        return totErr

    # restituisce il risultato della loss function calcolato per i pesi correnti e l'input (row,label)

    def forwardPropagation(self, row: tuple, label: tuple, hid_act_fun: str, out_act_fun: str, cost_fun: str):
        self.input_layer.output = np.asarray(row)
        # FIRST HIDDEN LAYER TAKES WEIGHTS FROM INPUT LAYER
        self.hidden_layers[0].feed_forward(
            self.input_layer.output, hid_act_fun)
        for pos, layer in enumerate(self.hidden_layers[1:]):
            layer.feed_forward(self.hidden_layers[pos].output, hid_act_fun)

        self.output_layer.feed_forward(
            self.hidden_layers[-1].output, act_fun=out_act_fun)

        if cost_fun == "mse":
            return mse(self.output_layer.output, np.array(label))
        elif cost_fun == "b_ce":
            return bin_cross_entropy(self.output_layer.output, np.array(label))
        else:
            return None

    def outLayerBackpropagation(self, label):
        # output layer
        self.output_layer.bias_gradients = np.zeros(
            self.output_layer.n_neurons)
        self.output_layer.weight_gradients = np.zeros(
            (self.output_layer.n_input, self.output_layer.n_neurons))
        # per ogni neurone del layer calcolo delta, gradiente dei pesi e gradiente del bias
        # considero delta del neurone il corrispettivo valore del gradiente (sono lo stesso valore)
        for i in np.arange(0, self.output_layer.n_neurons):
            '''
            i è l'i-esimo neurone del layer
            Per MSE delta_i = (d_i - o_i) * f'(net(i) ma f' = 1 => delta_i = (d_i - o_i)
            Per Cross-Entropy delta_i = [(d_i - o_i)/(o_i - o_i^2)] * f'(net(i)) ma f' = (o_i - o_i^2) => delta = (d_i - o_i)
            '''
            delta = label[i] - self.output_layer.output[i]
            self.output_layer.bias_gradients[i] = delta
            # gradiente_w_j,i = bias_i * o_j
            self.output_layer.weight_gradients[:,
            i] = delta * self.hidden_layers[-1].output

        self.output_layer.acc_bias_gradients += self.output_layer.bias_gradients
        self.output_layer.acc_weight_gradients += self.output_layer.weight_gradients

    def hiddenLayerBackpropagation(self, act_fun: str):
        for layer in np.arange(len(self.hidden_layers), 0, -1) - 1:
            if layer == len(self.hidden_layers) - 1:
                # ultimo hidden layer
                next_layer = self.output_layer
            else:
                next_layer = self.hidden_layers[layer + 1]
            if layer == 0:
                prec_layer = self.input_layer
            else:
                prec_layer = self.hidden_layers[layer - 1]
            layer = self.hidden_layers[layer]
            layer.bias_gradients = np.zeros(layer.n_neurons)
            layer.weight_gradients = np.zeros((layer.n_input, layer.n_neurons))
            # per ogni neurone del layer calcolo delta, gradiente dei pesi e gradiente del bias
            # considero delta del neurone il corrispettivo valore del gradiente (sono lo stesso valore)
            for i in np.arange(0, layer.n_neurons):
                # i è l'i-esimo neurone del layer
                # delta_i = (sommatoria(per tutti i neuroni k del layer successivo)(delta_k * w_i,k)* Drelu(net(i))
                sum = 0
                for k in np.arange(0, next_layer.n_neurons):
                    sum += next_layer.bias_gradients[k] * \
                           next_layer.weights[i][k]

                delta = sum * \
                        derivative(act_fun, (np.dot(prec_layer.output,
                                                    layer.weights[:, i]) + layer.biases[0][i]))
                layer.bias_gradients[i] = delta
                # gradiente_w_j,i = bias_i * o_j
                layer.weight_gradients[:, i] = delta * prec_layer.output

            layer.acc_bias_gradients += layer.bias_gradients
            layer.acc_weight_gradients += layer.weight_gradients

    # tengo due versioni di una matrice di gradienti per i pesi ed un vettore di gradienti per i bias
    # una versione è riservata ad i calcoli relativi ad uno specifico smple, mentre la versione "acc_" serve per il calolo del gradiente tenendo conto di tutti i samples

    def reset_accumulators(self):
        for layer in np.arange(len(self.hidden_layers), -1, -1) - 1:
            if layer == -1:
                layer = self.output_layer
            else:
                layer = self.hidden_layers[layer]
            layer.acc_bias_gradients = np.zeros(layer.n_neurons)
            layer.acc_weight_gradients = np.zeros(
                (layer.n_input, layer.n_neurons))


    def update_weights(self, n: int, eta,momentum=None, clip_value=None):
        for layer in np.arange(len(self.hidden_layers), -1, -1) - 1:
            if layer == -1:
                layer = self.output_layer
            else:
                layer = self.hidden_layers[layer]

            # clippo il gradiente per evitare gradient explosion
            if clip_value:
                layer.clip_gradients(clip_value)
            if momentum is None:
                layer.weights += eta * (layer.acc_weight_gradients / n)
                layer.biases[0] += eta * (layer.acc_bias_gradients / n)
            else:
                layer.weights += eta * (layer.acc_weight_gradients / n) + layer.momentum_velocity_w * momentum
                layer.biases[0] += eta * (layer.acc_bias_gradients / n) + layer.momentum_velocity_b * momentum
                # update velocities
                layer.momentum_velocity_w = eta * (layer.acc_weight_gradients / n) + layer.momentum_velocity_w * momentum
                layer.momentum_velocity_b = eta * (layer.acc_bias_gradients / n) + layer.momentum_velocity_b * momentum


    # n_mb -> minibatch size
    def train(self, df: pd.DataFrame,
              eta=None, epochs=1, clip_value=None, hid_act_fun: str = "None", out_act_fun: str = None,
              cost_fun: str = None, mb=16, momentum=None):

        n = df.shape[0]
        errors = []
        for epoch in np.arange(1, epochs + 1):
            # shuffle dataframe before each epoch
            df = df.sample(frac=1)

            for step in np.arange(0, n / mb):
                # tra uno step e l'altro azzero gli accumulatori dei gradienti per ogni layer
                self.reset_accumulators()
                # preparazione mb_Dataframe
                start_pos = int(step * mb)
                end_pos = int(start_pos + mb - 1)
                if end_pos >= n:
                    end_pos = int(n - 1)
                labels = df[["Class"]].iloc[start_pos:end_pos, :]
                data = (df.drop(["Class", "ID"], axis=1)).iloc[start_pos:end_pos, :]

                for row, label in zip(data.itertuples(index=False, name=None),
                                      labels.itertuples(index=False, name=None)):
                    # Forward propagation
                    self.forwardPropagation(
                        row, label, hid_act_fun, out_act_fun, cost_fun)
                    # backPropagation
                    self.outLayerBackpropagation(label)
                    # hidden layers
                    self.hiddenLayerBackpropagation(hid_act_fun)

                # aggiornamento dei pesi
                self.update_weights(mb, eta, momentum,clip_value)

                # new Total error with MSE
                # debug per clipping
                tot_err = self.calcError(
                    data, labels, hid_act_fun, out_act_fun, cost_fun)
                print(f"Epoch = {epoch}, step = {(int(step + 1))} total Error post-training = {tot_err}")
                if tot_err > 10000:
                    print(self)

            errors.append(tot_err)
            # end epoch

        print("end Training")
        return errors

    '''Return the number of layers of the network'''

    def __len__(self):
        res = 0
        if self.input_layer is not None:
            res += 1

        if self.hidden_layers is not None:
            res += len(self.hidden_layers)

        if self.output_layer is not None:
            res += 1

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
