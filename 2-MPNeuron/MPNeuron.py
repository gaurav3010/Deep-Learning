import sklearn.datasets
import numpy as np
import pandas as pd

breast_cancer = sklearn.datasets.load_breast_cancer()

X = breast_cancer.data
y = breast_cancer.target

data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
data['class'] = breast_cancer.target

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

"""
import matplotlib.pyplot as plt
plt.plot(X_train.T, '*')
plt.xticks(rotation='vertical')
plt.show()
"""

X_binarised_train = X_train.apply(pd.cut, bins=2, labels=[1, 0])
"""
plt.plot(X_binarised_train.T, '*')
plt.xticks(rotation='vertical')
plt.show()
"""

X_binarised_test = X_test.apply(pd.cut, bins=2, labels=[1, 0])
"""
plt.plot(X_binarised_test.T, '*')
plt.xticks(rotation='vertical')
plt.show()
"""

X_binarised_train = X_binarised_train.values
X_binarised_test = X_binarised_test.values

"""
b = 3
Y_pred_train = []
accurate_row = 0

for x, y in zip(X_binarised_train, Y_train):
    y_pred = (np.sum(x) >= b)
    Y_pred_train.append(y_pred)
    accurate_row += (y == y_pred)
    
print(accurate_row, accurate_row/X_binarised_train.shape[0])
"""

for b in range(X_binarised_train.shape[1] + 1):
    Y_pred_train = []
    accurate_row = 0

    for x, y in zip(X_binarised_train, Y_train):
        y_pred = (np.sum(x) >= b)
        Y_pred_train.append(y_pred)
        accurate_row += (y == y_pred)
        
    print(b, accurate_row, accurate_row/X_binarised_train.shape[0])
    
from sklearn.metrics import accuracy_score
b = 28

Y_pred_test = []

for x in X_binarised_test:
    y_pred = (np.sum(x) >= b)
    Y_pred_test.append(y_pred)
accuracy = accuracy_score(Y_pred_test, Y_test)
print(b, accuracy)

class MPNeuron:
    def __init__(self):
        self.b = None
    
    def model(self, x):
        return (sum(x) > self.b)
    
    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)
        
    def fit(self, X, Y):
        accuracy = {}
        
        for b in range(X.shape[1] + 1):
            self.b = b
            Y_pred = self.predict(X)
            accuracy[b] = accuracy_score(Y_pred, Y)
        best_b = max(accuracy, key=accuracy.get)
        self.b = best_b
        print('Optimal value of b is ', best_b)
        print('Highest accuracy is ', accuracy[best_b])
        
mp_neuron = MPNeuron()
mp_neuron.fit(X_binarised_train, Y_train)

Y_test_pred = mp_neuron.predict(X_binarised_test)
accuracy_test = accuracy_score(Y_test_pred, Y_test)

print(Y_test_pred)
print(accuracy_test)













        
            
























