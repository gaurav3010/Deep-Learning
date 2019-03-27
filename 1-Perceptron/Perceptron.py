import sklearn.datasets
import numpy as np
import pandas as pd


import sklearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

breast_cancer = sklearn.datasets.load_breast_cancer()

X = breast_cancer.data
y = breast_cancer.target

data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
data['class'] = breast_cancer.target

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

class Perceptron:
    def __init__(self):
        self.w = None
        self.b = None
        
    def model(self, x):
        return 1 if (np.dot(self.w, x) >= self.b) else 0
        
    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)
             
    def fit(self, X, Y, epochs = 2, lr = 1):
        self.w = np.ones(X.shape[1])
        self.b = 0
        
        accuracy = {}
        max_accuracy = 0
        
        for i in range(epochs):
            for x,y in zip(X, Y):
                y_pred = self.model(x)
                
                if y == 1 and y_pred == 0:
                    self.w = self.w + lr * x
                    self.b = self.b + lr * 1
                elif y ==0 and y_pred == 1:
                    self.w = self.w - lr * x
                    self.b = self.b - lr * 1
            accuracy[i] = accuracy_score(self.predict(X), Y)
            if(accuracy[i] > max_accuracy):
                max_accuracy = accuracy[i]
                chkptw = self.w
                chkptb = self.b
                
        self.w = chkptw
        self.b = chkptb
        #print(self.w)
        #print(self.b)
        plt.plot(accuracy.values())
        print(max_accuracy)
        
                
                

perceptron = Perceptron()

X_train = X_train.values
X_test = X_test.values

perceptron.fit(X_train, Y_train, 10000, 0.001)

from sklearn.metrics import accuracy_score
Y_pred_train = perceptron.predict(X_train)
print(accuracy_score(Y_pred_train, Y_train))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        