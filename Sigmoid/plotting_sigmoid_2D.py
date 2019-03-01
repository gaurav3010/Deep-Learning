import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, w, b):
    answer = 1/(1 + np.exp(-(w*x + b)))
    return answer

#print(sigmoid(1, 0.5, 0))
    
w = 4
b = 8
X = np.linspace(-10, 10, 100)
Y = sigmoid(X, w, b)
plt.plot(X, Y)
plt.show()