import numpy as np
import pandas as pd
from math_utils import *
from Layer import *
from tqdm import tqdm 

# ignore warnings for division by zero, invalid values and overflow
np.seterr(divide='ignore')
np.seterr(invalid='ignore')
np.seterr(over='ignore')

class NeuralNetwork:
    '''
        NeuralNetwork class contains:
        type: {"monk", "cup"}, the type of NN
        input_layer : Layer object
        hidden_layers: list of Layer objects
        output_layer: Layer object
    '''

    def __init__(self, input_layer: Layer = None, hidden_layers: list = None, output_layer: Layer = None, type="monk"):

        self.type = type
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers

        if hidden_layers is None:
            self.hidden_layers = []
        self.output_layer = output_layer

        return

    def add_input_layer(self, n_neurons: int = 0, randomize_weights=True):
        self.input_layer = Layer(
            n_inputs=1, n_neurons=n_neurons, is_input=True, randomize_weights=randomize_weights, type=self.type)
        return

    ''' 
        Add a new layer in the network. If the position is not specified, it is appended.
        The layer is created with the number of weights for each neuron relative to the previous layer
    '''

    def add_hidden_layer(self, n_inputs: int = 0, n_neurons: int = 0, pos: int = -1, randomize_weights=True):
        layer = Layer(n_inputs, n_neurons, is_input=False,
                      randomize_weights=randomize_weights, type=self.type)
        if pos == -1:
            self.hidden_layers.append(layer)

        else:
            self.hidden_layers.insert(pos, layer)
        return

    def add_output_layer(self, n_inputs: int = 0, n_neurons: int = 0, randomize_weights=True):
        self.output_layer = Layer(
            n_inputs, n_neurons, is_input=False, randomize_weights=randomize_weights)

        return

    def calcError(self, data: pd.DataFrame, labels: pd.DataFrame,
                  hid_act_fun: str = None, out_act_fun: str = None, cost_fun: list = None):
        # calcola l'errore sul dataset con i pesi e i bias correnti

        totErr = [0.0 for _ in range(len(cost_fun))]
        for row, label in zip(data.itertuples(index=False, name=None), labels.itertuples(index=False, name=None)):
            totErr = np.add(totErr, self.forwardPropagation(row, label, hid_act_fun, 
                                                            out_act_fun, cost_fun))
        if (data.shape[0]) != 0:
            totErr = np.divide(totErr,data.shape[0])
        if len(totErr) == 1:
            [totErr] = totErr
        return totErr
    '''
    The method propagates the input 'row' in the network.
    It returns the error (cost_fun) calculate between the output of the propagated input and 'label'
    '''
    def forwardPropagation(self, row: tuple, label: tuple, hid_act_fun: str, out_act_fun: str, cost_fun: list):

        self.input_layer.output = np.asarray(row)
        # FIRST HIDDEN LAYER TAKES WEIGHTS FROM INPUT LAYER
        self.hidden_layers[0].feed_forward(
            self.input_layer.output, hid_act_fun)
        for pos, layer in enumerate(self.hidden_layers[1:]):
            layer.feed_forward(self.hidden_layers[pos].output, hid_act_fun)

        self.output_layer.feed_forward(
            self.hidden_layers[-1].output, act_fun=out_act_fun)
        err = []
        for i in range(len(cost_fun)):
            if cost_fun[i] == "mse":
                err.append(mse(self.output_layer.output, np.array(label)))
            elif cost_fun[i] == "accuracy":
                err.append(accuracy(self.output_layer.output, np.array(label)))
            elif cost_fun[i] == "eucl":
                err.append(eucl(self.output_layer.output, np.array(label)))
            else:                
                return None

        return err

    # output layer backpropagation
    def outLayerBackpropagation(self, label, activationFunc):
        # output layer
        self.output_layer.bias_gradients = np.zeros(
            self.output_layer.n_neurons)
        self.output_layer.weight_gradients = np.zeros(
            (self.output_layer.n_inputs, self.output_layer.n_neurons))
        for i in np.arange(0, self.output_layer.n_neurons):
            '''
            i is the i-th layer neuron
            MSE delta_i = (d_i - o_i) * f'(net(i)) 
            '''
            net = np.dot(
                self.hidden_layers[-1].output, self.output_layer.weights[:, i]) + self.output_layer.biases[0][i]
            delta = (label[i] - self.output_layer.output[i]) * \
            derivative(activationFunc, net)
            self.output_layer.bias_gradients[i] = delta
            # gradient_w_j,i = bias_i * o_j
            self.output_layer.weight_gradients[:,
                                               i] = delta * self.hidden_layers[-1].output

        self.output_layer.acc_bias_gradients += self.output_layer.bias_gradients
        self.output_layer.acc_weight_gradients += self.output_layer.weight_gradients

    # hidden layers backpropagation
    def hiddenLayerBackpropagation(self, act_fun: str):
        for layer in np.arange(len(self.hidden_layers), 0, -1) - 1:
            # setting references to the precedent and next hidden layer, with the respect to the current one
            if layer == len(self.hidden_layers) - 1:
                # last hidden layer
                next_layer = self.output_layer
            else:
                next_layer = self.hidden_layers[layer + 1]
            if layer == 0:
                prec_layer = self.input_layer
            else:
                prec_layer = self.hidden_layers[layer - 1]

            layer = self.hidden_layers[layer]

            # weight and bias gradient calculation
            layer.bias_gradients= np.dot(next_layer.bias_gradients, next_layer.weights.T)*derivative(act_fun, (np.dot(prec_layer.output,
                                                                                                                      layer.weights) + layer.biases[0]))

            layer.weight_gradients = np.dot(prec_layer.output.reshape((len(prec_layer.output), 1)),
                                            layer.bias_gradients.reshape((1, len(layer.bias_gradients))))
            
            # the accumulator version, for each layer accumulate the gradients across the differents sample within the same updating step
            layer.acc_bias_gradients += layer.bias_gradients
            layer.acc_weight_gradients += layer.weight_gradients

    def reset_accumulators(self):
        for layer in np.arange(len(self.hidden_layers), -1, -1) - 1:
            if layer == -1:
                layer = self.output_layer
            else:
                layer = self.hidden_layers[layer]
            layer.acc_bias_gradients = np.zeros(layer.n_neurons)
            layer.acc_weight_gradients = np.zeros(
                (layer.n_inputs, layer.n_neurons))

    def update_weights(self, n: int, eta, momentum, ridge_lambda, lasso_lambda, clip_value=None, eta0=0,
                       decay_max_steps=None, decay_min_value=None, step=0):

        if decay_max_steps is not None:
            decay_alpha = step / decay_max_steps
            eta = eta0 * (1 - decay_alpha) + \
                (decay_alpha * eta0 / decay_min_value)
        if lasso_lambda is None and ridge_lambda is None:
            ridge_lambda = 0
        for layer in np.arange(len(self.hidden_layers), -1, -1) - 1:
            if layer == -1:
                layer = self.output_layer
            else:
                layer = self.hidden_layers[layer]

            if clip_value:
                layer.clip_gradients(clip_value)
            if ridge_lambda is not None:
                # REG l2
                if momentum is None:
                    layer.weights += eta * \
                        (layer.acc_weight_gradients / n) - \
                        (2 * ridge_lambda * layer.weights)
                    layer.biases[0] += eta * (layer.acc_bias_gradients / n)
                else:
                    deltaW = eta * (layer.acc_weight_gradients / n) + \
                        layer.momentum_velocity_w * momentum - \
                        (2 * ridge_lambda * layer.weights)
                    deltaB = eta * (layer.acc_bias_gradients / n) + \
                        layer.momentum_velocity_b * momentum
                    layer.weights += deltaW
                    layer.biases[0] += deltaB

                    # update velocities
                    layer.momentum_velocity_w = deltaW
                    layer.momentum_velocity_b = deltaB
            else:
                # REG l1
                '''
                the lambda (l1 reg term) sign on the weights update, depends on the weight sign
                (-lambda) if weight>=0, (+lambda) otherwise
                '''
                lasso_lambda_matrix = lasso_lambda * np.where(
                    layer.weights >= 0, -1, 1)
                if momentum is None:
                    layer.weights += eta * \
                        (layer.acc_weight_gradients / n) + lasso_lambda_matrix
                    layer.biases[0] += eta * (layer.acc_bias_gradients / n)
                else:
                    deltaW = eta * (layer.acc_weight_gradients / n) + \
                        layer.momentum_velocity_w * momentum + lasso_lambda_matrix
                    deltaB = eta * (layer.acc_bias_gradients / n) + \
                        layer.momentum_velocity_b * momentum
                    layer.weights += deltaW
                    layer.biases[0] += deltaB

                    # update velocities
                    # In this implementation, we choose not to include the reg terms on the momentum velocity 
                    # so that the reg term is independents to eta and alpha (momentum)
                    layer.momentum_velocity_w = deltaW
                    layer.momentum_velocity_b = deltaB

    
    def train(self, tr_data: pd.DataFrame, params, test_data=None, outFun2: str = None, type = None, es_data=None,es_stop= None, progress_bar=True):
        mb = params["mb"]
        epochs = params["epochs"]
        hid_act_fun = params["hid_act_fun"]
        out_act_fun = params["out_act_fun"]
        cost_fun = params["cost_fun"]
        eta = params["eta"]
        momentum = params["momentum"]
        clip_value = params["clip_value"]
        ridge_lambda = params["ridge_lambda"]
        lasso_lambda = params["lasso_lambda"]
        if params["decay_min_value"] is not None and params["decay_max_steps"] is not None:
            linear_decay = True
            decay_max_steps = params["decay_max_steps"]
            decay_step = 0
            eta0 = eta
            decay_min_value = params["decay_min_value"]
        else:
            linear_decay = False

        if "ID" in tr_data:
            tr_data.drop(["ID"], axis=1, inplace=True)

        n = tr_data.shape[0]
        train_errors = []
        if outFun2 is not None:
            fun2_train_err = []
        if test_data is not None:
            if "ID" in test_data:
                test_data.drop(["ID"], axis=1, inplace=True)
            test_errors = []
            if outFun2 is not None:
                fun2_test_err = []
        if es_data is not None:
            es_label_ = es_data[['TARGET_x', 'TARGET_y','TARGET_z']]
            es_data_ = es_data.drop(
                ['TARGET_x', 'TARGET_y', 'TARGET_z'], axis=1)
            min_esError=np.inf
            min_trError=np.inf
            min_testError=np.inf
            es_patience=params["es_patience"]
            epochsCounter=1
        
        # print every 10 seconds / 100 iterations
        for epoch in tqdm(np.arange(1, epochs + 1), desc="Training", unit="epoch", miniters=100, mininterval=10, disable=not progress_bar):
            # shuffle dataframe before each epoch
            
            tr_data = tr_data.sample(frac=1)
            if linear_decay and decay_step < decay_max_steps:
                decay_step += 1

            for step in np.arange(0, n / mb):
                # between each steps accumulators must be reset
                self.reset_accumulators()

                start_pos = int(step * mb)
                end_pos = int(start_pos + mb)
                if end_pos >= n:
                    end_pos = int(n)
                if type == "monk":
                    labels = tr_data[["Class"]].iloc[start_pos:end_pos, :]
                    data = (tr_data.drop(["Class"], axis=1)
                            ).iloc[start_pos:end_pos, :]
                else:
                    labels = tr_data[['TARGET_x', 'TARGET_y',
                                      'TARGET_z']].iloc[start_pos:end_pos, :]
                    data = (tr_data.drop(
                        ['TARGET_x', 'TARGET_y', 'TARGET_z'], axis=1)).iloc[start_pos:end_pos, :]

                for row, label in zip(data.itertuples(index=False, name=None),
                                      labels.itertuples(index=False, name=None)):
                    # Forward propagation
                    self.forwardPropagation(
                        row, label, hid_act_fun, out_act_fun, [cost_fun])
                    # backPropagation
                    self.outLayerBackpropagation(label, out_act_fun)
                    # hidden layers
                    self.hiddenLayerBackpropagation(hid_act_fun)
                # weights update
                if linear_decay:
                    self.update_weights(mb, eta, momentum, ridge_lambda, lasso_lambda,
                                        eta0=eta0, decay_max_steps=decay_max_steps, step=decay_step,
                                        decay_min_value=decay_min_value)
                else:
                    self.update_weights(mb, eta, momentum,
                                        ridge_lambda, lasso_lambda, clip_value)

            # at the end of each epochs errors are calculated
            if type == "monk":
                labels = tr_data[["Class"]]
                data = tr_data.drop(["Class"], axis=1)
            else:
                labels = tr_data[['TARGET_x', 'TARGET_y',
                                  'TARGET_z']]
                data = (tr_data.drop(
                    ['TARGET_x', 'TARGET_y', 'TARGET_z'], axis=1))

            if outFun2 is not None:
                [err1,err2] = self.calcError(
                    data, labels, hid_act_fun, out_act_fun, [cost_fun,outFun2])
                train_errors.append(err1)
                fun2_train_err.append(err2)
                if test_data is not None:
                    if type == "monk":
                        labels = test_data[["Class"]]
                        data = test_data.drop(["Class"], axis=1)
                    else:
                        labels = test_data[['TARGET_x', 'TARGET_y',
                                            'TARGET_z']]
                        data = test_data.drop(
                            ['TARGET_x', 'TARGET_y', 'TARGET_z'], axis=1)
                    [err1,err2] = self.calcError(
                        data, labels, hid_act_fun, out_act_fun, [cost_fun,outFun2])
                    test_errors.append(err1)
                    fun2_test_err.append(err2)
                    # trigger Early stopping if current training mee (fun2) < es_stop 
                    # only used when calculating both mee and mse (cost_fun) with training and test set in final retraining
                    if (es_stop is not None) and fun2_train_err[-1] < es_stop:
                            print(f"ES with tr_MEE below {fun2_train_err[-1]}")
                            return test_errors, train_errors, fun2_test_err, fun2_train_err

            else:
                train_errors.append(self.calcError(
                    data, labels, hid_act_fun, out_act_fun, [cost_fun]))
                if test_data is not None:
                    if type == "monk":
                        labels = test_data[["Class"]]
                        data = test_data.drop(["Class"], axis=1)
                    else:
                        labels = test_data[['TARGET_x', 'TARGET_y',
                                            'TARGET_z']]
                        data = test_data.drop(
                            ['TARGET_x', 'TARGET_y', 'TARGET_z'], axis=1)
                    error = self.calcError(
                        data, labels, hid_act_fun, out_act_fun, [cost_fun])
                    test_errors.append(error)
                    # trigger Early stopping if current training mee (cost_fun) < es_stop 
                    # only used when calculating both train and validation mee (cost_fun) on k-fold
                    if (es_stop is not None) and train_errors[-1] < es_stop:
                            print(f"ES with tr_MEE below {train_errors[-1]}")
                            return test_errors[-1], train_errors[-1]

                    

            if es_data is not None and test_data is not None:
                esError = self.calcError(
                    es_data_, es_label_, hid_act_fun, out_act_fun, [cost_fun])
             
                if np.isnan(esError):
                    # overflow
                    return min_testError, min_trError
                if esError > min_esError:
                    if epochsCounter > es_patience:
                        # Stop training
                        print("EARLY STOPPED")
                        return  min_testError, min_trError
                    else:
                        epochsCounter+= 1
                if esError < min_esError:
                    min_esError = esError
                    min_trError = train_errors[-1]
                    min_testError = test_errors[-1]
                    epochsCounter = 1

            # end epoch
        # end training

        if es_data is not None and test_data is not None:
            return  min_testError, min_trError
        if outFun2 is not None and test_data is not None:
            return test_errors, train_errors, fun2_test_err, fun2_train_err
        elif outFun2 is None and test_data is not None:
            return test_errors, train_errors
        elif outFun2 is not None and test_data is None:
            return train_errors, fun2_train_err

        return train_errors
    

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
