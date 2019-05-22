# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:48:38 2019

@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
import seaborn as sns
from tqdm import trange

from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris

from numpy.linalg import norm

my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
np.random.seed(0)

iris = load_iris()
data = iris.data[:, :2]
labels = iris.target

plt.scatter(data[:, 0], data[:, 1], c = labels, cmap=my_cmap)
plt.show()

print("Data shape : ", data.shape)
print("Labels shape : ", labels.shape)

X_train, X_val, Y_train, Y_val = train_test_split(data, labels, stratify=labels, random_state=0, test_size=0.2)
print(X_train.shape, X_val.shape, labels.shape)

enc = OneHotEncoder()
Y_OH_train = enc.fit_transform(np.expand_dims(Y_train, 1)).toarray()
Y_OH_val = enc.fit_transform(np.expand_dims(Y_val, 1)).toarray()
print(Y_OH_train.shape, Y_OH_val.shape)


class FFNetwork:
    
    def __init__(self, num_hidden = 2, init_method = 'xavier', activation_function = 'sigmoid', leaky_slope = 0.1):
        self.params = {}
        #self.params_h = []
        self.num_layers = 2
        self.layer_size = [2, num_hidden, 3]
        self.activation_function = activation_function
        self.leaky_slope = leaky_slope
        np.random.seed(0)
        
        if init_method == "random":
            for i in range(1, self.num_layers+1):
                self.params["W"+str(i)] = np.random.rand(self.layer_size[i-1], self.layer_size[i])
                self.params["B"+str(i)] = np.random.rand(1,self.layer_size[i])
        
        elif  init_method == "he":
            for i in range(1, self.num_layers+1):
                self.params["W"+str(i)] = np.random.rand(self.layer_size[i-1], self.layer_size[i]) * np.sqrt(2/self.layer_size[i-1])
                self.params["B"+str(i)] = np.random.rand(1,self.layer_size[i])
                
        elif init_method == "xavier":
            for i in range(1, self.num_layers+1):
                self.params["W"+str(i)] = np.random.rand(self.layer_size[i-1], self.layer_size[i]) * np.sqrt(1/self.layer_size[i-1])
                self.params["B"+str(i)] = np.random.rand(1,self.layer_size[i])
                
        elif init_method == "zeros":
            for i in range(1, self.num_layers+1):
                self.params["W"+str(i)] = np.zeros((self.layer_size[i-1], self.layer_size[i]))
                self.params["B"+str(i)] = np.zeros((1,self.layer_size[i]))
                
        self.gradients = {}
        self.update_params = {}
        self.prev_update_params = {}
        for i in range(1, self.num_layers+1):
            self.update_params["v_w"+str(i)] = 0
            self.update_params["v_b"+str(i)] = 0
            self.update_params["m_w"+str(i)] = 0
            self.update_params["m_b"+str(i)] = 0
            self.prev_update_params["v_w"+str(i)] = 0
            self.prev_update_params["v_b"+str(i)] = 0
            
            
    def forward_activation(self, X):
        if self.activation_function == "sigmoid":
            return 1.0 / (1.0 + np.exp(-X))
        elif self.activation_function == "tanh":
            return np.tanh(X)
        elif self.activation_function == "relu":
            return np.maximum(0, X)
        elif self.activation_function == "leaky_relu":
            return np.maximum(self.leaky_relu*X, X)
        
    def grad_activation(self, X):
        if self.activation_function == "sigmoid":
            return X*(1 - X)
        elif self.activation_function == "tanh":
            return (1 - np.square(X))
        elif self.activation_function == "relu":
            return 1.0 * (X>0)
        elif self.activation_function == "leaky_relu":
            d = np.zeros_like(X)
            d[X<=0] = self.leaky_slope
            d[d>0] = 1
            return d    
        
    def get_accuracy(self):
        Y_pred_train = model.predict(X_train)
        Y_pred_train = np.argmax(Y_pred_train, 1)
        Y_pred_val = model.predict(X_val)
        Y_pred_val = np.argmax(Y_pred_val, 1)
        accuracy_train = accuracy_score(Y_pred_train, Y_train)
        accuracy_val = accuracy_score(Y_pred_val, Y_val)
        return accuracy_train, accuracy_val
    
    def softmax(self, X):
        exps = np.exp(X)
        return exps / np.sum(exps, axis=1).reshape(-1, 1)
    
    def forward_pass(self, X, params=None):
        if params is None:
            params = self.params
        self.A1 = np.matmul(X, params["W1"]) + params["B1"] # (N, 2) * (2, 2) -> (N, 2)
        self.H1 = self.forward_activation(self.A1) # (N, 2)
        self.A2 = np.matmul(self.H1, params["W2"]) + params["B2"] # (N, 2) * (2, 4) -> (N, 4)
        self.H2 = self.softmax(self.A2) # (N, 4)
        return self.H2
    
    def grad(self, X, Y, params=None):
        if params is None:
            params = self.params
            
        self.forward_pass(X, params)
        m = X.shape[0]
        #print(self.H2.shape, Y.shape)
        self.gradients["dA2"] = self.H2 - Y # (N, 4) - (N, 4) -> (N, 4)
        self.gradients["dW2"] = np.matmul(self.H1.T, self.gradients["dA2"]) # (2, N) * (N, 4)
        self.gradients["dB2"] = np.sum(self.gradients["dA2"], axis=0).reshape(1, -1) # (N, 4) -> (1, 4)
        self.gradients["dH1"] = np.matmul(self.gradients["dA2"], params["W2"].T) # (N ,4) * (4, 2) -> (N, 2)
        self.gradients["dA1"] = np.multiply(self.gradients["dH1"], self.grad_activation(self.H1)) # (N, 2) .* (N, 2) -> (N, 2)
        self.gradients["dW1"] = np.matmul(X.T, self.gradients["dA1"]) # (2, N) * (N, 2) -> (2, 2)
        self.gradients["dB1"] = np.sum(self.gradients["dA1"], axis=0).reshape(1, -1) # (N, 2) -> (1, 2)
            
    def fit(self, X, Y, epochs=1, algo="GD", l2_norms=False, lambda_val=0.8, display_loss=False, eta=1):
        train_accuracy = {}
        val_accuracy = {}
        if display_loss:
            loss = []
            weight_mag = []
            
        for num_epoch in trange(epochs):
            m = X.shape[0]
            
            self.grad(X, Y)
            for i in range(1, self.num_layers+1):
                if l2_norms:
                    self.params["W"+str(i)] -= (eta * lambda_val)/m * self.params["W"+str(i)] + eta * (self.gradients["dW"+str(i)]/m)
                else:
                    self.params["W"+str(i)] -= eta * (self.gradients["dW"+str(i)]/m)
                self.params["B"+str(i)] -= eta * (self.gradients["dB"+str(i)]/m)
                
            train_acc, val_acc = self.get_accuracy()
            train_accuracy[num_epoch] = train_acc
            val_accuracy[num_epoch] = val_acc
            
            if display_loss:
                Y_pred = self.predict(X)
                loss.append(log_loss(np.argmax(Y, axis=1), Y_pred))
                weight_mag.append((norm(self.params["W1"]) + norm(self.params["W2"]) + norm(self.params["B1"]) + norm(self.params["B2"]))/18)
                
        plt.plot(train_accuracy.values(), label='Train_accuracy')
        plt.plot(val_accuracy.values(), label='Validation accuracy')
        plt.plot(np.ones((epochs, 1)) * 0.9)
        plt.plot(np.ones((epochs, 1)) * 0.33)
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.show()
        
        if display_loss:
            print(weight_mag)
            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.set_xlabel("epochs")
            ax1.set_ylabel("Log Loss", color=color)
            ax1.plot(loss, '-o', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel("Weight Magnitude", color=color)
            ax2.plot(weight_mag, '-*', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            fig.tight_layout()
            plt.show()
            
    def predict(self, X):
        Y_pred = self.forward_pass(X)
        return np.array(Y_pred).squeeze()
    
    
def print_accuracy():
    Y_pred_train = model.predict(X_train)
    Y_pred_train = np.argmax(Y_pred_train, 1)
    Y_pred_val = model.predict(X_val)
    Y_pred_val = np.argmax(Y_pred_val, 1)
    accuracy_train = accuracy_score(Y_pred_train, Y_train)
    accuracy_val = accuracy_score(Y_pred_val, Y_val)
    print("Train accuracy : ", accuracy_train)
    print("Validation accuracy : ", accuracy_val)
    
model = FFNetwork(num_hidden=1)
model.fit(X_train, Y_OH_train, epochs=100, eta=0.5)
print_accuracy()

model = FFNetwork(num_hidden=2)
model.fit(X_train, Y_OH_train, epochs=100, eta=0.5)
print_accuracy()

model = FFNetwork(num_hidden=4)
model.fit(X_train, Y_OH_train, epochs=500, eta=0.1)
print_accuracy()

model = FFNetwork(num_hidden=8)
model.fit(X_train, Y_OH_train, epochs=400, eta=0.2)
print_accuracy()

model = FFNetwork(num_hidden=16)
model.fit(X_train, Y_OH_train, epochs=1300, eta=0.2)
print_accuracy()

model = FFNetwork(num_hidden=32)
model.fit(X_train, Y_OH_train, epochs=1000, eta=0.1)
print_accuracy()

model = FFNetwork(num_hidden=64)
model.fit(X_train, Y_OH_train, epochs=1000, eta=0.1)
print_accuracy()

""" ------------- Add Regularisation --------------- """

model = FFNetwork(num_hidden=2)
model.fit(X_train, Y_OH_train, epochs=2000, eta=0.1, l2_norms=True, lambda_val=0.1, display_loss=True)
print_accuracy()

model = FFNetwork(num_hidden=2)
model.fit(X_train, Y_OH_train, epochs=2000, eta=0.1, l2_norms=True, lambda_val=1, display_loss=True)
print_accuracy()

model = FFNetwork(num_hidden=2)
model.fit(X_train, Y_OH_train, epochs=2000, eta=0.5, l2_norms=True, lambda_val=5, display_loss=True)
print_accuracy()
            
            
""" ----------- Adding Noise ------------ """

for noise_fraction in [0.01, 0.05, 0.1, 0.15, 0.18, 0.2]:
    print(noise_fraction)
    X_train_noisy = X_train * (1 - noise_fraction*np.random.rand(X_train.shape[0], X_train.shape[1]))
    model = FFNetwork(num_hidden=64)
    model.fit(X_train, Y_OH_train, epochs=2000, eta=0.1)
    print_accuracy()
    
""" ------------ Early Stopping ------------ """

model = FFNetwork(num_hidden=64)
model.fit(X_train, Y_OH_train, epochs=100, eta=0.1)
print_accuracy()
        
model = FFNetwork(num_hidden=64)
model.fit(X_train, Y_OH_train, epochs=500, eta=0.1)
print_accuracy()
        
        