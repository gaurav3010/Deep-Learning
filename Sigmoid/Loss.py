import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, w, b):
    return 1/(1 + np.exp(-(w*x + b)))

def calculate_loss(X, Y, w_est, b_est):
    loss = 0
    for x, y in zip(X, Y):
        loss = loss + (y - sigmoid(x, w_est, b_est)) ** 2
    return loss
        
w_unknow = 0.5
b_unknow = 0.25

X = np.random.random(25) * 20 - 10
Y = sigmoid(X, w_unknow, b_unknow)

plt.plot(X, Y, "*")
plt.show()


W = np.linspace(-1, 1, 100)
B = np.linspace(-1, 1, 100)

WW, BB = np.meshgrid(W, B)
Loss = np.zeros(WW.shape)
#print(WW.shape)

for i in range(WW.shape[0]):
    for j in range(WW.shape[1]):
        Loss[i, j] = calculate_loss(X, Y, WW[i, j], BB[i, j]) 
        
ij = np.argmin(Loss)
i = int(np.floor(ij/Loss.shape[1]))
j = int(ij - i * Loss.shape[1])

print(i, j)

print(WW[i, j], BB[i, j])