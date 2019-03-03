import numpy as np
import matplotlib.pyplot as plt

class SigmoidNeuron:
    
    def __init__(self):
        self.w = None
        self.b = None
    
    def perceptron(self, x):
        return np.dot(x, self.w.T) + self.b
    
    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))
    
    def grad_w(self, x, y):
        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred - y) * y_pred * (1 - y_pred) * x
    
    def grad_b(self, x, y):
        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred - y) * y_pred * (1 - y_pred)
    
    def fit(self, X, Y, epochs = 1, learning_rate = 1):
        self.w = np.random.rand(1, X.shape[1])
        self.b = 0
        
        for i in range(epochs):
            dw = 0
            db = 0
            for x, y in zip(X, Y):
                dw += self.grad_w(x, y)
                db += self.grad_b(x, y)
            self.w -= learning_rate * dw
            self.b -= learning_rate * db
            
X = np.asarray([[2.5, 2.5], [4, -1], [1, -4], [3, 1.25], [2, 4], [1, 5]])
Y = [1, 1, 1, 0, 0, 0]

sn = SigmoidNeuron()
sn.fit(X, Y, 1, 0.25)
        
print(sn.w, sn.b)

for i in range(10):
    sn.fit(X, Y, 1, 0.25)
    print(sn.w, sn.b)