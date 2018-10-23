
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import random
import numpy as np

class FullNetLayer(object):

    def __init__(self, x_len, y_len, activator):
 
        self.x = np.zeros((x_len, 1))
        self.y = np.zeros((y_len, 1))
        self.W = np.random.uniform(-1.0, 1.0, (y_len, x_len))
        self.b = np.zeros((y_len, 1))

        self.activator = activator

    def forward(self, x):

        self.x = x
        
        self.y = self.activator.forward(np.dot(self.W, self.x) + self.b)
        

    def backward(self, delta):

        self.delta = self.activator.backward(self.x) * np.dot(self.W.T, delta)
        
        self.W_grad = np.dot(delta, self.x.T)
        self.b_grad =  delta

    def update(self, rate):

        self.W += rate * self.W_grad
        self.b += rate * self.b_grad


class SigmoidActivator(object):

    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)


class MPLNet(object):

    def __init__(self, layers):

        self.layers = []
        
        for i in range(len(layers)-1):
            self.layers.append(FullNetLayer(layers[i], layers[i+1], SigmoidActivator()))
    
    def train(self, data_set, labels, rate, epoch):
        
        for _ in range(epoch):
            for x, y in zip(data_set, labels):
                x_mat = x.reshape(len(x), 1)
                y_mat = y.reshape(len(y), 1)
                self.train_one_sample(x_mat, y_mat, rate)

    def train_one_sample(self, x, y, rate):

        self.predict(x)
        self.calc_grad(y)
        self.update_w(rate)

    def predict(self, x):

        data = x
        for layer in self.layers:
            layer.forward(data)
            data = layer.y

        return data

    def calc_grad(self, t):

        delta = self.layers[-1].activator.backward(self.layers[-1].y) * (t - self.layers[-1].y)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_w(self, rate):
        for layer in self.layers:
            layer.update(rate)

if __name__ == "__main__":

    mpl = MPLNet([2, 4, 1])

    data_set = np.array([[0 ,0], [1, 0], [0, 1], [1, 1]])
    labels = np.array(([0], [1], [1], [0]))
    
   

    mpl.train(data_set, labels, 0.1,10000)
  
    data = np.array([0, 0])
    data.resize(2, 1)
    print(mpl.predict(data))