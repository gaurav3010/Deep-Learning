# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 23:46:54 2019

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm import trange

from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs

# Firest Feed Forward Neural Network

class FFNetwork1:
    def __init__(self):
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.w3 = np.random.randn()
        self.w4 = np.random.randn()
        self.w5 = np.random.randn()
        self.w6 = np.random.randn()
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        
    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def forward_pass(self, x):
        self.x1, self.x2 = x
        self.a1 = self.w1*self.x1 + self.w2*self.x2 + self.b1
        self.h1 = self.sigmoid(self.a1)
        self.a2 = self.w3*self.x1 + self.w4*self.x2 + self.b2
        self.h2 = self.sigmoid(self.a2)
        self.a3 = self.w5*self.h1 + self.w6*self.h2 + self.b3
        self.h3 = self.sigmoid(self.a3)
        return self.h3
    
    def grad(self, x, y):
        self.forward_pass(x)
        self.dw1 = (self.h3-y) * self.h3*(1-self.h3) * self.w5 * self.h1*(1-self.h1) * self.x1
    
    def fit(self, X, Y, epochs=1, learning_rate=1, initialize=True, display_loss=False):
      
        if display_loss:
            loss = {}
            w1 = {}
        
        for i in trange(epochs):
            dw1, dw2, dw3, dw4, dw5, dw6, db1, db2, db3 = [0]*9 
            
            for x, y in zip(X, Y):
                self.grad(x, y)
                dw1 += self.dw1
                
            m = X.shape[1]
            
            self.w1 -= learning_rate * dw1 / m 
                
            if display_loss:
                w1[i] = self.w1
                Y_pred = self.predict(X)
                loss[i] = mean_squared_error(Y_pred, Y)
            
        if display_loss:
            plt.tight_layout()
            plt.subplot(2, 1, 1)
            plt.plot(w1.values())
            plt.xlabel('Epochs')
            plt.ylabel('W1')
            
            """plt.plot(2, 1, 2)
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            plt.show()"""
                
    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred)
    
# Generate Data

data, labels = make_blobs(n_samples = 1000, centers = 4, n_features = 2, random_state = 0)
print(data.shape, labels.shape)

plt.scatter(data[:, 0], data[:, 1], c = labels)
plt.show()
    
labels_orig = labels
labels = np.mod(labels_orig, 2)


X_train, X_val, Y_train, Y_val = train_test_split(data, labels, stratify = labels, random_state = 0)
print(X_train.shape, X_val.shape)

ffn1 = FFNetwork1()
ffn1.fit(X_train, Y_train, epochs = 500, learning_rate = 1, display_loss = True)

    
    
""" ---------------------- All weight update ------------------------- """

# Firest Feed Forward Neural Network

class FirstFFNetwork:
    def __init__(self):
        np.random.seed(0)
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.w3 = np.random.randn()
        self.w4 = np.random.randn()
        self.w5 = np.random.randn()
        self.w6 = np.random.randn()
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        
    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def forward_pass(self, x):
        self.x1, self.x2 = x
        self.a1 = self.w1*self.x1 + self.w2*self.x2 + self.b1
        self.h1 = self.sigmoid(self.a1)
        self.a2 = self.w3*self.x1 + self.w4*self.x2 + self.b2
        self.h2 = self.sigmoid(self.a2)
        self.a3 = self.w5*self.h1 + self.w6*self.h2 + self.b3
        self.h3 = self.sigmoid(self.a3)
        return self.h3

    
    def grad(self, x, y):
        self.forward_pass(x)
        
        self.dw5 = (self.h3-y) * self.h3*(1-self.h3) * self.h1
        self.dw6 = (self.h3-y) * self.h3*(1-self.h3) * self.h2
        self.db3 = (self.h3-y) * self.h3*(1-self.h3)
        
        self.dw1 = (self.h3-y) * self.h3*(1-self.h3) * self.w5 * self.h1*(1-self.h1) * self.x1
        self.dw2 = (self.h3-y) * self.h3*(1-self.h3) * self.w5 * self.h1*(1-self.h1) * self.x2
        self.db1 = (self.h3-y) * self.h3*(1-self.h3) * self.w5 * self.h1*(1-self.h1)
        
        self.dw3 = (self.h3-y) * self.h3*(1-self.h3) * self.w6 * self.h2*(1-self.h2) * self.x1
        self.dw4 = (self.h3-y) * self.h3*(1-self.h3) * self.w6 * self.h2*(1-self.h2) * self.x2
        self.db2 = (self.h3-y) * self.h3*(1-self.h3) * self.w6 * self.h2*(1-self.h2)
    
        
    def fit(self, X, Y, epochs=1, learning_rate=1, initialize=True, display_loss=False, display_weight=False):
        
        if initialize:
            np.random.seed(0)
            self.w1 = np.random.randn()
            self.w2 = np.random.randn()
            self.w3 = np.random.randn()
            self.w4 = np.random.randn()
            self.w5 = np.random.randn()
            self.w6 = np.random.randn()
            self.b1 = 0
            self.b2 = 0
            self.b3 = 0
            
        if display_loss:
            loss = {}
        
        for i in trange(epochs):
            dw1, dw2, dw3, dw4, dw5, dw6, db1, db2, db3 = [0]*9 
            
            for x, y in zip(X, Y):
                self.grad(x, y)
                dw1 += self.dw1
                dw2 += self.dw2
                dw3 += self.dw3
                dw4 += self.dw4
                dw5 += self.dw5
                dw6 += self.dw6
                db1 += self.db1
                db2 += self.db2
                db3 += self.db3
                
            m = X.shape[0]
            
            self.w1 -= learning_rate * dw1 / m 
            self.w2 -= learning_rate * dw2 / m 
            self.w3 -= learning_rate * dw3 / m 
            self.w4 -= learning_rate * dw4 / m 
            self.w5 -= learning_rate * dw5 / m 
            self.w6 -= learning_rate * dw6 / m 
            self.b1 -= learning_rate * db1 / m
            self.b2 -= learning_rate * db2 / m
            self.b3 -= learning_rate * db3 / m
                
            if display_loss:
                Y_pred = self.predict(X)
                loss[i] = mean_squared_error(Y_pred, Y)
                
            if display_weight:
                weight_matrix = np.array([[0, self.b3, self.w5, self.w6, 0, 0], [self.b1, self.w1, self.w2, self.b2, self.w3, self.w4]])
                weight_matrices.append(weight_matrix)
            
        if display_loss:
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            plt.show()
                
    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred)
       
    def predict_h1(self, X, binary=True):
        Y_pred = []
        for x in X:
          y_pred = self.forward_pass(x)
          Y_pred.append(self.h1)
        return np.array(Y_pred)
    
        
    
    def predict_h2(self, X, binary=True):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(self.h2)
        return np.array(Y_pred)
    
    def predict_h3(self, X, binary=True):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(self.h3)
        return np.array(Y_pred)
        
weight_matrices = []
ffn = FirstFFNetwork()
ffn.fit(X_train, Y_train, epochs=2000, learning_rate=5, display_loss=True, display_weight=True)
    
    
""" ------------------------ Visualize a each neuron --------------------------- """

def make_meshgrid(x, y, h=0.2):
    x_min, x_max = x.min() - 0.5, x.max() + 0.5
    y_min, y_max = y.min() - 0.5, y.max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, predict, xx, yy, **params):
    Z = predict(np.c_[xx.ravel(), yy.ravel()])
    Z =Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
    
def plot_boundary():
    xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1])
    predict_functions = [ffn.predict_h1, ffn.predict_h2, ffn.predict_h3]
    
    for i in range(3):
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_contours(ax, predict_functions[i], xx, yy, alpha=0.2)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, alpha=0.2)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title("h"+str(i+1))
    
    return True

plot_boundary()
    
    
    
    
    
    
    
    
    
    