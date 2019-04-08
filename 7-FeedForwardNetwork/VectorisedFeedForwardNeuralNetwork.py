import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from tqdm import trange

from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs

import time


# Generate Data

data, labels = make_blobs(n_samples = 1000, centers = 4, n_features = 2, random_state = 0)
print(data.shape, labels.shape)

plt.scatter(data[:, 0], data[:, 1], c = labels)
plt.show()

labels_orig = labels
labels = np.mod(labels_orig, 2)

plt.scatter(data[:, 0], data[:, 1], c = labels)
plt.show()

X_train, X_val, Y_train, Y_val = train_test_split(data, labels_orig, stratify = labels_orig, random_state = 0)
print(X_train.shape, X_val.shape, labels_orig.shape)

enc = OneHotEncoder()
y_OH_train = enc.fit_transform(np.expand_dims(Y_train, 1)).toarray()
y_OH_val = enc.fit_transform(np.expand_dims(Y_val, 1)).toarray()
print(y_OH_train.shape, y_OH_val.shape)

W1 = np.random.randn(2, 2)
W2 = np.random.randn(2, 4)
print(W1)
print(W2)

""" -------------------- Weight Vectorised Version ------------------------- """

class FF_MultiClass_WeightVectorised:
    def __init__(self, W1, W2):
        self.W1 = W1.copy()
        self.W2 = W2.copy()
        self.B1 = np.zeros((1, 2))
        self.B2 = np.zeros((1, 4))
        
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def softmax(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps)
    
    def forward_pass(self, x):
        x = x.reshape(1, -1) # (1, 2)
        self.A1 = np.matmul(x, self.W1) + self.B1 # (1, 2)*(2, 2) --> (1, 2)
        self.H1 = self.sigmoid(self.A1) # (1, 2)
        self.A2 = np.matmul(self.H1, self.W2) + self.B2 # (1, 2)*(2, 4) --> (1, 4)
        self.H2 = self.softmax(self.A2) # (1, 4)
        return self.H2
    
    def grad_sigmoid(self, x):
        return x*(1-x)
    
    def grad(self, x, y):
        self.forward_pass(x)
        x = x.reshape(1, -1) # (1, 2)
        y = y.reshape(1, -1) # (1, 4)
        
        self.dA2 = self.H2 - y # (1, 2)
        
        self.dW2 = np.matmul(self.H1.T, self.dA2) # (2, 1) * (1, 4) --> (2, 4)
        self.dB2 = self.dA2 # (1, 4)
        self.dH1 = np.matmul(self.dA2, self.W2.T) # (1, 4) * (4, 2) --> (1, 2)
        self.dA1 = np.multiply(self.dH1, self.grad_sigmoid(self.H1)) # (1 ,2)
        
        self.dW1 = np.matmul(x.T, self.dA1) # (2, 1) * (1, 2) --> (2, 2)
        self.dB1 = self.dA1 # (1, 2)
        
    def fit(self, X, Y, epochs=1, learning_rate=1, display_loss=False):
         
        if display_loss:
            loss = {}
            
        for i in trange(epochs):
            dW1 = np.zeros((2, 2))
            dW2 = np.zeros((2, 4))
            dB1 = np.zeros((1, 2))
            dB2 = np.zeros((1, 4))
            
            for x, y in zip(X, Y):
                self.grad(x, y)
                dW1 += self.dW1
                dW2 += self.dW2
                dB1 += self.dB1 
                dB2 += self.dB2
                
            m = x.shape[0]
            self.W2 -= learning_rate * (dW2/m)
            self.B2 -= learning_rate * (dB2/m)
            self.W1 -= learning_rate * (dW1/m)
            self.B1 -= learning_rate * (dB1/m)
            
            if display_loss:
                Y_pred = self.predict(X)
                loss[i] = log_loss(np.argmax(Y, axis=1), Y_pred)
                
        if display_loss:
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('Log Loss')
            plt.show()
    
    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred).squeeze()
    
    
""" ---------------------- Input + Weight Vectorised Version ----------------------- """

class FF_MultiClass_InputWeightVectorised:
    def __init__(self, W1, W2):
        self.W1 = W1.copy()
        self.W2 = W2.copy()
        self.B1 = np.zeros((1, 2))
        self.B2 = np.zeros((1, 4))
        
    def sigmoid(self, X):
        return 1.0/(1.0 + np.exp(-X))
    
    def softmax(self, X):
        exps = np.exp(X)
        return exps / np.sum(exps, axis=1).reshape(-1, 1)
    
    def forward_pass(self, X):
        self.A1 = np.matmul(X, self.W1) + self.B1 # (N, 2) * (2, 2) --> (N, 2)
        self.H1 = self.sigmoid(self.A1)  # (N, 2)
        self.A2 = np.matmul(self.H1, self.W2) + self.B2 # (N, 2) * (2, 4) --> (N, 4)
        self.H2 = self.softmax(self.A2) # (N, 4)
        return self.H2
    
    def grad_sigmoid(self, X):
        return X * (1 - X)
    
    def grad(self, X, Y):
        self.forward_pass(X)
        m = X.shape[0]
        
        self.dA2 = self.H2 - Y # (N, 4) - (N, 4)
        
        self.dW2 = np.matmul(self.H1.T, self.dA2) # (2, N) * (N, 4) --> (2, 4)
        self.dB2 = np.sum(self.A2, axis=0).reshape(1, -1) # (N, 4) --> (1, 4)
        self.dH1 = np.matmul(self.dA2, self.W2.T) # (N, 4) * (4, 2) --> (N, 2)
        self.dA1 = np.multiply(self.dH1, self.grad_sigmoid(self.H1)) # (N, 2) .* (N, 2) --> (N, 2)
        
        self.dW1 = np.matmul(X.T, self.dA1) # (2, N) * (N, 2) --> (2, 2)
        self.dB1 = np.sum(self.dA1, axis=0).reshape(1, -1) # (N, 2) --> (1, 2)
        
    def fit(self, X, Y, epochs=1, learning_rate=1, display_loss=False):
         
        if display_loss:
            loss = {}
            
        for i in trange(epochs):
            self.grad(X, Y)
                
            m = X.shape[0]
            self.W2 -= learning_rate * (self.dW2/m)
            self.B2 -= learning_rate * (self.dB2/m)
            self.W1 -= learning_rate * (self.dW1/m)
            self.B1 -= learning_rate * (self.dB1/m)
            
            if display_loss:
                Y_pred = self.predict(X)
                loss[i] = log_loss(np.argmax(Y, axis=1), Y_pred)
                
        if display_loss:
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('Log Loss')
            plt.show()
            
    def predict(self, X):
        Y_pred = self.forward_pass(X)
        return np.array(Y_pred).squeeze()
    
""" ------------------------ Compare Three Class base on Time -------------------- """

model_init = [FF_MultiClass_WeightVectorised(W1, W2),FF_MultiClass_InputWeightVectorised(W1, W2)]
models = []
for idx, model in enumerate(model_init, start=1):
    tic = time.time()
    ffsn_multi_specific = model
    ffsn_multi_specific.fit(X_train, y_OH_train, epochs=2000, learning_rate=0.5, display_loss=True)
    models.append(ffsn_multi_specific)
    toc = time.time()
    print("Time taken by model {}: {}".format(idx, toc-tic))
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


















