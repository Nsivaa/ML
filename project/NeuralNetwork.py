import numpy as np
import pandas as pd
from math_utils import *
from Layer import *


class NeuralNetwork:
    '''
        NeuralNetwork class contains:
        input_layer : Layer object
        hidden_layers: list of Layer objects
        output_layer: Layer object
    '''

    def __init__(self, input_layer: Layer = None, hidden_layers: list = None, output_layer: Layer = None,type="monk"):

        self.type=type
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers

        if hidden_layers is None:
            self.hidden_layers = []
        self.output_layer = output_layer

        return

    def add_input_layer(self, n_neurons: int = 0, randomize_weights=True):
        self.input_layer = Layer(
            n_inputs=1, n_neurons=n_neurons, is_input=True, randomize_weights=randomize_weights,type=self.type)
        return

    ''' 
        Add a new layer in the network. If the position is not specified, it is appended.
        The layer is created with the number of weights for each neuron relative to the previous layer
    '''

    def add_hidden_layer(self, n_inputs: int = 0, n_neurons: int = 0, pos: int = -1, randomize_weights=True):
        layer = Layer(n_inputs, n_neurons, is_input=False,
                      randomize_weights=randomize_weights,type=self.type)
        if pos == -1:
            self.hidden_layers.append(layer)

        else:
            self.hidden_layers.insert(pos, layer)
        return

    def add_output_layer(self, n_inputs: int = 0, n_neurons: int = 0, randomize_weights=True):
        self.output_layer = Layer(
            n_inputs, n_neurons, is_input=False, randomize_weights=randomize_weights,type=self.type)

        return

    def calcError(self, data: pd.DataFrame, labels: pd.DataFrame,
                  hid_act_fun: str = None, out_act_fun: str = None, cost_fun: str = None):
        # calcola l'errore sul dataset con i pesi e i bias correnti

        totErr = 0
        for row, label in zip(data.itertuples(index=False, name=None), labels.itertuples(index=False, name=None)):
            totErr += self.forwardPropagation(row, label, hid_act_fun,
                                              out_act_fun, cost_fun)
        if (data.shape[0]) != 0:
            totErr /= data.shape[0]

        return totErr

    def forwardPropagation(self, row: tuple, label: tuple, hid_act_fun: str, out_act_fun: str, cost_fun: str):
        # restituisce il risultato della loss function calcolato per i pesi correnti e l'input (row,label)

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
        elif cost_fun == "accuracy":
            return accuracy(self.output_layer.output, np.array(label))
        elif cost_fun == "eucl":
            return eucl(self.output_layer.output, np.array(label))
        else:
            return None

    def outLayerBackpropagation(self, label, activationFunc):
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
            Per MSE delta_i = (d_i - o_i) * f'(net(i)) 
            Per Cross-Entropy delta_i = [(d_i - o_i)/(o_i - o_i^2)] * f'(net(i)) ma f' = (o_i - o_i^2) => delta = (d_i - o_i)
            '''
            net = np.dot(self.hidden_layers[-1].output, self.output_layer.weights[:, i]) + self.output_layer.biases[0][i]
            delta = (label[i] - self.output_layer.output[i]) * derivative(activationFunc, net)
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
                # delta_i = (sommatoria(per tutti i neuroni k del layer successivo)(delta_k * w_i,k)* D(net(i))
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

            # clippo il gradiente per evitare gradient explosion
            if clip_value:
                layer.clip_gradients(clip_value)
            if ridge_lambda is not None:
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
                # nella formula del gradiente con lasso regression si ha la derivata del valore assoluto dei pesi
                # perciò la derivata, che si sommerà ad eta e al momentum nel calcolo del nuovo peso sarà:
                # -lambda per peso >=0 mentre + lambda altrimenti
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
                    layer.momentum_velocity_w = deltaW
                    layer.momentum_velocity_b = deltaB

    # n_mb -> minibatch size
    # TODO: il parametro di regolarizzazione si è scelto di non considerarlo nell'aggiornamento della velocità del momentum così da mantenerlo indipendente da eta e alpha (momentum)
    '''
        Nella versione con ridge regression, viene  agggiunto il termine lambda * norma(2) al quadrato del vettore 
        contenente tutti i pesi del network alla funzione di loss e, di conseguenza anche all'aggiornamento dei pesi,
        L'errore che si stampa nei plot e che si usa per i confronti tra i modelli è quello classico, che non tiene conto della regolarizzazione
        il termine di regolarizzazaione sarà solo aggiunto nel weight update. 
        Lambda dovrebbe esser moltiplicato per mb/n ma si evita in quanto il parametro sarò automaticamente selezionato nelle grid search
    '''
    #l'esecuzione attesa di train con ES implica la restituzione di due soli valori (da utilizzare er retrain senza ES), anzichè di 1 o 2 o 4 liste di errori (da utilizzare per il plot)
    #test_data != None => si calcola anche validation/test loss, outFun2 != None => si calcolano anche traing e test Error
    def train(self, tr_data: pd.DataFrame, params, test_data=None, outFun2: str = None, type = None,es_data=None):
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
            es_data_ = es_data.drop(['TARGET_x', 'TARGET_y', 'TARGET_z'], axis=1)
            min_esError=10000
            min_trError=10000
            min_testError=10000
            es_patience=params["es_patience"]
            epochsCounter=1
            #se usaimo ES, continua ad iterare finchè

        #TODO: stabilire se far continuare all'inifito o finche non overfitta nel caso in cui si attiva ES'
        for epoch in np.arange(1, epochs + 1):
            # shuffle dataframe before each epoch
            np.random.seed()
            tr_data = tr_data.sample(frac=1)
            if linear_decay and decay_step < decay_max_steps:
                decay_step += 1

            for step in np.arange(0, n / mb):
                # tra uno step e l'altro azzero gli accumulatori dei gradienti per ogni layer
                self.reset_accumulators()
                # preparazione mb_Dataframe
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
                        row, label, hid_act_fun, out_act_fun, cost_fun)
                    # backPropagation
                    self.outLayerBackpropagation(label, out_act_fun)
                    # hidden layers
                    self.hiddenLayerBackpropagation(hid_act_fun)

                # aggiornamento dei pesi
                if linear_decay:
                    self.update_weights(mb, eta, momentum, ridge_lambda, lasso_lambda,
                                        eta0=eta0, decay_max_steps=decay_max_steps, step=decay_step,
                                        decay_min_value=decay_min_value)
                else:
                    self.update_weights(mb, eta, momentum,
                                        ridge_lambda, lasso_lambda, clip_value)

            # calcolo  degli error su test e fun2
            if type == "monk":
                labels = tr_data[["Class"]]
                data = tr_data.drop(["Class"], axis=1)
            else:
                labels = tr_data[['TARGET_x', 'TARGET_y',
                                  'TARGET_z']]
                data = (tr_data.drop(
                    ['TARGET_x', 'TARGET_y', 'TARGET_z'], axis=1))
            train_errors.append(self.calcError(
                data, labels, hid_act_fun, out_act_fun, cost_fun))

            #Early stopping non va applicato anche su fun2
            if outFun2 is not None:
                fun2_train_err.append(self.calcError(
                    data, labels, hid_act_fun, out_act_fun, outFun2))
                if test_data is not None:
                    if type == "monk":
                        labels = test_data[["Class"]]
                        data = test_data.drop(["Class"], axis=1)
                    else:
                        labels = test_data[['TARGET_x', 'TARGET_y',
                                            'TARGET_z']]
                        data = test_data.drop(
                            ['TARGET_x', 'TARGET_y', 'TARGET_z'], axis=1)
                    test_errors.append(self.calcError(
                        data, labels, hid_act_fun, out_act_fun, cost_fun))
                    fun2_test_err.append(self.calcError(
                        data, labels, hid_act_fun, out_act_fun, outFun2))
            elif test_data is not None:
                if type == "monk":
                    labels = test_data[["Class"]]
                    data = test_data.drop(["Class"], axis=1)
                else:
                    labels = test_data[['TARGET_x', 'TARGET_y',
                                      'TARGET_z']]
                    data = test_data.drop(
                        ['TARGET_x', 'TARGET_y', 'TARGET_z'], axis=1)
                test_errors.append(self.calcError(
                    data, labels, hid_act_fun, out_act_fun, cost_fun))

            if es_data is not None and test_data is not None:
                esError= self.calcError(
                            es_data_, es_label_, hid_act_fun, out_act_fun, cost_fun)
                if esError > min_esError:
                    if epochsCounter > es_patience:
                        #Stop training
                        return min_trError, min_testError
                    else:
                        epochsCounter+=1
                if esError < min_esError:
                    min_esError=esError
                    min_trError=train_errors[-1]
                    min_testError = test_errors[-1]
                    epochsCounter=1

            # end epoch
        # end training
        #print("end Training")
        if es_data is not None and test_data is not None:
            return min_trError, min_testError


        if outFun2 is not None and test_data is not None:
            return test_errors, train_errors, fun2_test_err, fun2_train_err
        elif outFun2 is None and test_data is not None:
            return test_errors, train_errors
        elif outFun2 is not None and test_data is None:
            return train_errors, fun2_train_err

        return train_errors
    #si suppone che la k-fold sia solo usata nella grid, che è solo usata in Cup con ES
    def k_fold(self, k, data, parameters,es_data):
        '''
        theta is the parameter we are performing the search on 
        parameters is the list of all other parameters
        values is the list of values to try for theta
        '''
        if "ID" in data.columns:
            data.drop(["ID"], axis=1, inplace=True)
        np.random.seed()
        data = data.sample(frac=1)
        folds = np.array_split(data, k)
        valid_errors=[]
        tr_errors=[]
        for fold in folds:
            tr_set = pd.concat(
                [f for f in folds if not (pd.Series.equals(f, fold))], axis=0)
            tr_error, valid_error= self.train(tr_set, parameters,test_data=fold,es_data=es_data)
            #print(tr_set)
            valid_errors.append(valid_error)
            tr_errors.append(tr_error)
        valid_errors=np.array(valid_errors)
        tr_errors= np.array(tr_errors)
        valid_mean=valid_errors.mean()
        tr_mean = tr_errors.mean()
        valid_var=valid_errors.var()
        #print(f"Valid error:{mean}, Variance = {var}, with par {parameters}")
        return tr_mean,valid_mean, valid_var

    def hold_out(self, data, parameters, randomize_shuffle=True):
        if "ID" in data.columns:
            data.drop(["ID"], axis=1, inplace=True)
        if randomize_shuffle:
            np.random.seed()
        else:
            np.random.seed(0)

        data = data.sample(frac=1)
        folds = np.array_split(data, 5)

        tr_set = pd.concat(folds[1:], axis=0)
        test_set = folds[-1]
        self.train(tr_set, parameters)
        valid_labels = test_set[["Class"]]
        valid_data = test_set.drop(["Class"], axis=1)
        valid_err = self.calcError(valid_data, valid_labels,
                                   parameters["hid_act_fun"], parameters["out_act_fun"],
                                   parameters["cost_fun"])
        print(f"Valid error:{valid_err} with par {parameters}")
        return valid_err

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
