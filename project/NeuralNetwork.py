import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from utils import *
from Layer import *

# TODO: cambiare bias in bias[0](in modo da avere un vettore e non una matrice con una riga) e modificare tutto di conseguenza


# TODO: CAMBIARE ACTIVATION FUNCTION OUTPUT LAYER CLASSIFICAZIONE

# np.random.seed(0)

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

    def update_weights(self, n: int, eta, momentum=None, clip_value=None):
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
                layer.weights += eta * \
                    (layer.acc_weight_gradients / n) + \
                    layer.momentum_velocity_w * momentum
                layer.biases[0] += eta * (layer.acc_bias_gradients / n) + \
                    layer.momentum_velocity_b * momentum
                
                # update velocities
                layer.momentum_velocity_w = eta * \
                    (layer.acc_weight_gradients / n) + \
                    layer.momentum_velocity_w * momentum
                layer.momentum_velocity_b = eta * \
                    (layer.acc_bias_gradients / n) + \
                    layer.momentum_velocity_b * momentum

    # n_mb -> minibatch size


    def train(self, tr_data: pd.DataFrame, params : dict):
        mb = params["mb"]
        epochs = params["epochs"]
        hid_act_fun = params["hid_act_fun"]
        out_act_fun = params["out_act_fun"]
        cost_fun = params["cost_fun"]
        eta = params["eta"]
        momentum = params["momentum"]
        clip_value = params["clip_value"]

        
        if "ID" in tr_data:
            tr_data.drop(["ID"], axis = 1, inplace=True)
        n = tr_data.shape[0]
        errors = []
        for epoch in np.arange(1, epochs + 1):
            # shuffle dataframe before each epoch
            tr_data = tr_data.sample(frac=1)

            for step in np.arange(0, n / mb):
                # tra uno step e l'altro azzero gli accumulatori dei gradienti per ogni layer
                self.reset_accumulators()
                # preparazione mb_Dataframe
                start_pos = int(step * mb)
                end_pos = int(start_pos + mb - 1)
                if end_pos >= n:
                    end_pos = int(n - 1)
                labels = tr_data[["Class"]].iloc[start_pos:end_pos, :]
                data = (tr_data.drop(["Class"], axis=1)
                        ).iloc[start_pos:end_pos, :]                
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
                self.update_weights(mb, eta, momentum, clip_value)

                # new Total error with MSE
                # debug per clipping
                tot_err = self.calcError(
                    data, labels, hid_act_fun, out_act_fun, cost_fun)
                print(
                    f"Epoch = {epoch}, step = {(int(step + 1))} total Error post-training = {tot_err}")
                if tot_err > 10000:
                    print(self)

            errors.append(tot_err)
            # end epoch

        print("end Training")
        return errors

    def grid_search(self, k, data, grid):
        np.random.seed(0)

        prod = product(grid)
       # for par in grid.keys():
            #self.k_fold(k, data, , par, par.get())

    def k_fold(self, k, data, parameters, theta, values): 
        '''
        theta is the parameter we are performing the search on 
        parameters is the list of all other parameters
        values is the list of values to try for theta
        '''
        val_errors={}
        if "ID" in data.columns:
            data.drop(["ID"], axis = 1, inplace=True)
        data = data.sample(frac=1)
        folds = np.array_split(data, k)
        for value in values:
            parameters[theta] = value
            valid_err_accumulator = 0

            for fold in folds:

                tr_set = pd.concat([f for f in folds if pd.Series.equals(f, fold)], axis=1)
                self.train(tr_set, parameters)
                valid_labels = fold[["Class"]]
                valid_data = fold.drop(["Class"], axis=1)
                valid_err_accumulator += self.calcError(valid_data, valid_labels, 
                                                        parameters["hid_act_fun"],parameters["out_act_fun"],
                                                        parameters["cost_fun"])

            val_errors[value] = valid_err_accumulator / k

        min_err = np.min(val_errors.values())
        return val_errors[min_err]


    def __len__(self):
        '''Return the number of layers of the network'''

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
