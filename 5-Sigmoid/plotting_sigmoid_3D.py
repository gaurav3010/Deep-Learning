import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def sigmoid(x1, x2, w1, w2, b):
    answer = 1/(1 + np.exp(-(w1*x1 + w2*x2 + b)))
    return answer

X1 = np.linspace(-10, 10, 100)
X2 = np.linspace(-10, 10, 100)

XX1, XX2 = np.meshgrid(X1, X2)

print(X1.shape, X2.shape, XX1.shape, XX2.shape)

w1 = 0.5
w2 = 0.5
b = 0

Y = sigmoid(XX1, XX2, w1, w2, b)
#print(Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(XX1, XX2, Y, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y');