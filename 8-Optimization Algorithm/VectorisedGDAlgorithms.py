import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
import seaborn as sns
from tqdm import trange

from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs

my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
np.random.seed(0)

data, labels = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=0)
print(data.shape, labels.shape)

plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=my_cmap)
plt.show()

labels_orig = labels
labels = np.mod(labels_orig, 2)

plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=my_cmap)
plt.show()

X_train, X_val, Y_train, Y_val =train_test_split(data, labels_orig, stratify=labels_orig, random_state=0)
print(X_train.shape, X_val.shape, labels_orig.shape)

enc = OneHotEncoder()
Y_OH_train = enc.fit_transform(np.expand_dims(Y_train, 1)).toarray()
Y_OH_val = enc.fit_transform(np.expand_dims(Y_val, 1)).toarray()
print(Y_OH_train.shape, Y_OH_val.shape)

W1 = np.random.randn(2, 2)
W2 = np.random.randn(2, 4)
print(W1)
print(W2)

"""   --------------- FF Class ------------------ """

class FFNetwork:
    
    def __init__(self, W1, W2):
        self.params = {}
        self.params["W1"] = W1.copy()
        self.params["W2"] = W2.copy()
        self.params["B1"] = np.zeros((1, 2))
        self.params["B2"] = np.zeros((1, 4))
        self.num_layers = 2
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
        return 1.0/(1.0 + np.exp(-X))
    
    def grad_activation(self, X):
        return X*(1 - X)
    
    def softmax(self, X):
        exps = np.exp(X)
        return exps / np.sum(exps, axis=1).reshpe(-1, 1)
    
    def forward_pass(self, X, params=None):
        if params is None:
            params = self.params
        self.A1 = np.matmul(X, params["W1"]) + params["B1"] # (N, 2) * (2, 2) -> (N, 2)
        self.H1 = self.forward_activation(self.A1) # (N, 2)
        self.A2 = np.matmul(self.H1, params["W2"]) + self.params["B2"] # (N, 2) * (2, 4) -> (N, 4)
        self.H2 = self.forward_activation(self.A2) # (N, 4)
        return self.H2
    
    def grad(self, X, Y, params=None):
        if params is None:
            params = self.params
        self.forward_pass(X, params)
        m = X.shape[0]
        self.gradients["dA2"] = self.H2 - Y # (N, 4) - (N, 4) -> (N, 4)
        self.gradients["dW2"] = np.matmul(self.H1.T, self.gradients["dA2"]) # (2, N) * (N, 4)
        self.gradients["dB2"] = np.sum(self.gradients["dA2"], axis=0).reshpae(-1, 1) # (N, 4) -> (1, 4)
        self.gradients["dH1"] = np.matmul(self.gradients["dA2"], params["W2"].T) # (N ,4) * (4, 2) -> (N, 2)
        self.gradients["dA1"] = np.multiply(self.gradients["dH1"], self.grad_activation(self.H1)) # (N, 2) .* (N, 2) -> (N, 2)
        self.gradients["dW1"] = np.matmul(X.T, self.gradients["dA1"]) # (2, N) * (N, 2) -> (2, 2)
        self.gradients["dB1"] = np.sum(self.gradients["dA1"], axis=0).reshape(-1, 1) # (N, 2) -> (1, 2)
        
    def fit(self, X, Y, epochs=1, algo="GD", display_loss=False, eta=1,
            mini_batch_size=100, eps=1e-8, beta=0.9, beta1=0.9,
            beta2=0.9, gamma=0.9):
        if display_loss:
            loss = {}
        
        for num_epoch in trange(epochs):
            m = X.shape[0]
            
            if algo == "GD":
                self.grad(X, Y)
                for i in range(1, self.num_layers+1):
                    self.params["W"+str(i)] -= eta * (self.gradients["dW"+str(i)]/m)
                    self.params["B"+str(i)] -= eta * (self.gradients["dB"+str(i)]/m)
                    
            if algo == "MiniBatch":
                for k in range(0, m, mini_batch_size):
                    self.grad(X[k:k+mini_batch_size], Y[k:k+mini_batch_size])
                    for i in range(1, self.num_layers+1):
                        self.params["W"+str(i)] -= eta * (self.gradients["dW"+str(i)]/mini_batch_size)
                        self.params["B"+str(i)] -= eta * (self.gradients["dB"+str(i)]/mini_batch_size)
                        
            if algo == "Momentum":
                for k in range(1, self.num_layers+1):
                    self.update_params["v_w"+str(i)] = gamma * self.update_params["v_w"+str(i)] + eta * (self.gradients["dW"+str(i)]/m)
                    self.update_params["v_b"+str(i)] = gamma * self.update_params["v_b"+str(i)] + eta * (self.gradients["dB"+str(i)]/m)
                    self.params["W"+str(i)] -= eta * (self.update_params["v_w"+str(i)])
                    self.params["B"+str(i)] -= eta * (self.gradients["v_b"+str(i)])
                      
            if algo == "NAG":
                temp_params = {}
                for i in range(1, self.num_layers+1):
                    self.update_params["v_w"+str(i)] = gamma * self.prev_update_params["v_w"+str(i)]
                    self.update_params["v_b"+str(i)] = gamma * self.prev_update_params["v_b"+str(i)]
                    temp_params["W"+str(i)] = self.params["W"+str(i)] - self.update_params["v_w"+str(i)]
                    temp_params["B"+str(i)] = self.params["B"+str(i)] - self.update_params["v_b"+str(i)]
                self.grad(X, Y, temp_params)
                for i in range(1, self.num_layers+1):
                    self.update_params["v_w"+str(i)] = gamma * self.update_params["v_w"+str(i)] + eta * self.gradients["dW"+str(i)]
                    self.update_params["v_b"+str(i)] = gamma * self.update_params["v_b"+str(i)] + eta * self.gradients["dB"+str(i)]
                    self.params["W"+str(i)] -= eta * self.update_params["v_w"+str(i)]
                    self.params["B"+str(i)] -= eta * self.update_params["v_b"+str(i)]
                self.prev_update_params = self.update_params
            
            if algo == "AdaGrad":
                self.grad(X, Y)
                for k in range(1, num_layers+1):
                    self.update_params["v_w"+str(i)] = beta * self.update_params["v_w"+str(i)] + (1 - beta) * self.gradients["W"+str(i)]
                    self.update_params["v_b"+str(i)] = beta * self.update_params["v_b"+str(i)] + (1 - beta) * self.gradients["B"+str(i)]
                    self.params["W"+str(i)] -= (eta / (np.sqrt(self.update_params["v_w"+str(i)]+eps))) * (self.gradients["dW"+str(i)]) 
                    self.params["B"+str(i)] -= (eta / (np.sqrt(self.update_params["v_b"+str(i)]+eps))) * (self.gradients["dB"+str(i)]) 
    
            

























 