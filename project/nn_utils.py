import numpy as np
import pandas as pd
import matplotlib as plt

'''If the value is greater than 0, we leave it as is, else we replace it with value * 0,01'''
def leaky_relu(x):
    if x > 0:
        return x
    return x * 0.01 

class Node:
    ''' 
        The Node class represents each vertex of the graph 
        The attribute value represents the stored data
        The list of neighbors attribute represents the vertices with which exists a connection 
    '''
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias        
    
    def __str__(self):    
        return f" weight: {self.weight}, bias: {self.bias}"

class Layer:
    '''
        The Layer class represents a layer of the neural network.
        It has a list of Nodes
    '''
    def __init__(self, nodes = None):
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes
    
    def add_node(self, node = None):
        if node is None:
            print(f'WARNING: Trying to add NONE Node')
        else:
            self.nodes.append(node)
            
    def __len__(self):
        return len(self.nodes)

    def __str__(self):
        out_str = ""
        for pos, node in enumerate(self.nodes):
            out_str += f"node {pos} -> " + str(node) + "\n"
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

   # def feed_forward():

