import pandas as pd 
import tqdm as tqdm
import numpy as np
from math_utils import *
from NeuralNetwork import *
class Ensemble:
    '''
        Class which model for the ensemble model of the best 5 models out of model evaluation using grid search
        Ensemble class contains:
        models : ensemble models
        plot_ensemble: plot utils 
        train_models: train all models on tr-set and test it on test-set, you can also add early stopping errors
        
    '''

    def __init__(self, structures,n=5):
        # models contains the 'n' best networks
        self.models = []
        # structures contains the number of layer and neurons of each of the ensemble models
        self.structures=structures
        self.n = n
        self.n_inputs = 10
        self.n_outputs = 3
        # the following are array of array containg the training\test errors
        self.tr_mse = [[],[],[],[],[]]
        self.tr_mee = [[],[],[],[],[]]
        self.test_mse = [[],[],[],[],[]]
        self.test_mee = [[],[],[],[],[]]
        # whenever es[i] became True, the i-th model must skip training on the remanining epochs
        self.ES = [False,False,False, False, False]
        
    def buildModels(self):
        for (layers, n_neurons) in self.structures:
            net = NeuralNetwork(type="cup")
            net.add_input_layer(self.n_inputs)
            net.add_hidden_layer(self.n_inputs, n_neurons)
            for i in range(layers-1):
                net.add_hidden_layer(n_neurons, n_neurons)
            net.add_output_layer(n_neurons, self.n_outputs)
            self.models.append(net)

    '''
    The method propagates the input 'row' for each network of the ensemble and then averages the results.
    It returns the error (cost_fun) calculated between the output of the propagated input and 'label'
    '''
    def forwardPropagation(self, row: tuple, label: tuple, hid_act_fun: str, out_act_fun: str, cost_fun: list, onlyPrediction=False):
        # 3 is the number of outputs of each model
        outputs = np.zeros(3,dtype=float)
        for i in range(self.n):
            self.models[i].input_layer.output = np.asarray(row)
            self.models[i].hidden_layers[0].feed_forward(
                self.models[i].input_layer.output, hid_act_fun)
            
            for pos, layer in enumerate(self.models[i].hidden_layers[1:]):
                layer.feed_forward(self.models[i].hidden_layers[pos].output, hid_act_fun)

            self.models[i].output_layer.feed_forward(
                self.models[i].hidden_layers[-1].output, act_fun=out_act_fun)
            outputs+=self.models[i].output_layer.output

        #now we average the predictions and then the error can be calculated
        outputs/=self.n
        if onlyPrediction:
            return outputs
        err = []
        for i in range(len(cost_fun)):
            if cost_fun[i] == "mse":
                err.append(mse(outputs, np.array(label)))
            else:
                err.append(eucl(outputs, np.array(label)))
        return err

    # identical method as the NeuralNetwork one
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
    train_params contains the train parameters of each model of the ensemble
    The training method is very similar to the NeuralNetwork one but it's written with some semplifications
    taking into account the different use cases
    '''
    def train_models(self, tr_data: pd.DataFrame, train_params, test_data=None, outFun2: str = None,epochs = 2000):
        # build models
        self.buildModels()
        mb = train_params["mb"]
        es_stop = train_params["es_stop"]
        eta = train_params["eta"]
        momentum = train_params["momentum"]
        ridge_lambda = train_params["ridge_lambda"]
        # the following three functions are not arrays
        hid_act_fun = train_params["hid_act_fun"]
        out_act_fun = train_params["out_act_fun"]
        cost_fun = train_params["cost_fun"]

        if "ID" in tr_data:
            tr_data.drop(["ID"], axis=1, inplace=True)

        n = tr_data.shape[0]
        if test_data is not None:
            if "ID" in test_data:
                test_data.drop(["ID"], axis=1, inplace=True)
        train_mee = []
        test_mee = []
        train_mse = []
        test_mse = []
        
        # print every 10 seconds / 100 iterations
        for epoch in tqdm(np.arange(1, epochs + 1), desc="Training", unit="epoch", miniters=100, mininterval=10):
            if all([earlyStopped == True for earlyStopped in self.ES]):
                # all models are early stopped so the training of the ensamble is completed
                break
            # train an epoch for each model
            tr_data = tr_data.sample(frac=1)
            for i in range(self.n):
                if self.ES[i]:
                    continue
                for step in np.arange(0, n / mb[i]):
                    self.models[i].reset_accumulators()
                    start_pos = int(step * mb[i])
                    end_pos = int(start_pos + mb[i])
                    if end_pos >= n:
                        end_pos = int(n)
                   
                    labels = tr_data[['TARGET_x', 'TARGET_y',
                                    'TARGET_z']].iloc[start_pos:end_pos, :]
                    data = (tr_data.drop(
                        ['TARGET_x', 'TARGET_y', 'TARGET_z'], axis=1)).iloc[start_pos:end_pos, :]

                    for row, label in zip(data.itertuples(index=False, name=None),
                                        labels.itertuples(index=False, name=None)):
                        # Forward propagation
                        self.models[i].forwardPropagation(
                            row, label, hid_act_fun, out_act_fun, [cost_fun])

                        # backPropagation
                        self.models[i].outLayerBackpropagation(label, out_act_fun)
                        # hidden layers
                        self.models[i].hiddenLayerBackpropagation(hid_act_fun)
                    # aggiornamento dei pesi
                    self.models[i].update_weights(mb[i], eta[i], momentum[i],
                                            ridge_lambda[i], None,None)

                # calcolo  degli error su test e fun2 a fine epoca
                labels = tr_data[['TARGET_x', 'TARGET_y',
                                    'TARGET_z']]
                data = (tr_data.drop(
                        ['TARGET_x', 'TARGET_y', 'TARGET_z'], axis=1))
                
                # test data is always not None
                if outFun2 is not None:
                    # use case: retraining -> cost fun = mse, fun2 = mee (eucl),               
                    [err1,err2] = self.models[i].calcError(
                        data, labels, hid_act_fun, out_act_fun, [cost_fun,outFun2])
                    self.tr_mse[i].append(err1)
                    self.tr_mee[i].append(err2)                
                    labels = test_data[['TARGET_x', 'TARGET_y',
                                        'TARGET_z']]
                    data = test_data.drop(
                        ['TARGET_x', 'TARGET_y', 'TARGET_z'], axis=1)
                    [err1,err2] = self.models[i].calcError(
                        data, labels, hid_act_fun, out_act_fun, [cost_fun,outFun2])
                    self.test_mse[i].append(err1)
                    self.test_mee[i].append(err2)
                    # trigger Early stopping if current training mee (fun2) < es_stop 
                    # only used when calculating both mee and mse (cost_fun) with training and test set in final retraining
                    if (es_stop[i] is not None) and self.tr_mee[i][-1] < es_stop[i]:
                            print(f"ES with tr_MEE below {self.tr_mee[i][-1]}")
                            self.ES[i] =True
                else:
                    # use case: k-fold -> cost fun = mee     
                    self.tr_mee[i].append(self.models[i].calcError(
                        data, labels, hid_act_fun, out_act_fun, [cost_fun]))                       
                    labels = test_data[['TARGET_x', 'TARGET_y',
                                        'TARGET_z']]
                    data = test_data.drop(
                        ['TARGET_x', 'TARGET_y', 'TARGET_z'], axis=1)
                    error = self.models[i].calcError(
                        data, labels, hid_act_fun, out_act_fun, [cost_fun])
                    self.test_mee[i].append(error)
                    # trigger Early stopping if current training mee (cost_fun) < es_stop 
                    # only used when calculating both train and validation mee (cost_fun) on k-fold
                    if (es_stop[i] is not None) and self.tr_mee[i][-1] < es_stop[i]:
                            print(f"ES with tr_MEE below {self.tr_mee[i][-1]}")
                            self.ES[i] =True
                # end i-th model
            # ensamble error
            labels = tr_data[['TARGET_x', 'TARGET_y',
                                'TARGET_z']]
            data = (tr_data.drop(
                    ['TARGET_x', 'TARGET_y', 'TARGET_z'], axis=1))

            
            if outFun2 is None:
                train_mee.append(self.calcError(
                    data, labels, hid_act_fun, out_act_fun, [cost_fun]))
                # val set
                labels = test_data[['TARGET_x', 'TARGET_y',
                                        'TARGET_z']]
                data = test_data.drop(
                        ['TARGET_x', 'TARGET_y', 'TARGET_z'], axis=1)
                test_mee.append(self.calcError(
                    data, labels, hid_act_fun, out_act_fun, [cost_fun]))
            else:
                [mse,mee] = self.calcError(
                    data, labels, hid_act_fun, out_act_fun, [cost_fun,outFun2])
                train_mse.append(mse)
                train_mee.append(mee)
                labels = test_data[['TARGET_x', 'TARGET_y',
                                        'TARGET_z']]
                data = test_data.drop(
                        ['TARGET_x', 'TARGET_y', 'TARGET_z'], axis=1)
                [mse,mee] = self.calcError(
                    data, labels, hid_act_fun, out_act_fun, [cost_fun,outFun2])
                test_mse.append(mse)
                test_mee.append(mee)
            # end epoch
        # end training
        if outFun2 is None:
            #k-fold
            return test_mee[-1], train_mee[-1]
        else: 
            # retraining
            return test_mse, train_mse, test_mee, train_mee

    
    def __str__(self):
        out = ""
        for model in self.models:
            out+=str(model)+"\n"
        return out
