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

data, labels = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=0)
print(data.shape, labels.shape)

plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=my_cmap)
plt.show()

labels_orig = labels
labels = np.mod(labels_orig, 2)

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


"""   --------------- FF Class ------------------ """

class FFNetwork:
    
    def __init__(self, init_method = 'random', activation_function = 'sigmoid', leaky_slope = 0.1):
        self.params = {}
        self.params_h = []
        self.num_layers = 2
        self.layer_size = [2, 2, 4]
        self.activation_function = activation_function
        self.leaky_slope = leaky_slope
        
        np.random.speed(0)
        
        if init_method == "random":
            for i in range(1, self.num_layers+1):
                self.params["W"+str(i)] = np.random.rand(self.layers_sizes[i-1], self.layers_sizes[i])
                self.params["B"+str(i)] = np.random.rand(1,self.layers_sizes[i])
        
        elif  init_method == "he":
            for i in range(1, self.num_layers+1):
                self.params["W"+str(i)] = np.random.rand(self.layers_sizes[i-1], self.layers_sizes[i]) * np.sqrt(2/self.layers_sizes[i-1])
                self.params["B"+str(i)] = np.random.rand(1,self.layers_sizes[i])
                
        elif init_method == "xavier":
            for i in range(1, self.num_layers+1):
                self.params["W"+str(i)] = np.random.rand(self.layers_sizes[i-1], self.layers_sizes[i]) * np.sqrt(1/self.layers_sizes[i-1])
                self.params["B"+str(i)] = np.random.rand(1,self.layers_sizes[i])
                
        elif init_method == "zeros":
            for i in range(1, self.num_layers+1):
                self.params["W"+str(i)] = np.zeros((self.layers_sizes[i-1], self.layers_sizes[i]))
                self.params["B"+str(i)] = np.zeros((1,self.layers_sizes[i]))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
