import os
import sys
import pickle
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import trange
from sklearn.metrics import confusion_matrix
import csv

#np.random.seed(100)

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
    
    def fit(self, X, Y, epochs = 1, learning_rate = 1, display_loss=False):
        self.w = np.random.rand(1, X.shape[1])
        self.b = 0
        
        if display_loss:
            loss = {}
        
        for i in trange(epochs):
            dw = 0
            db = 0
            for x, y in zip(X, Y):
                dw += self.grad_w(x, y)
                db += self.grad_b(x, y)
            self.w -= learning_rate * dw
            self.b -= learning_rate * db
            
            if display_loss:
                Y_pred = self.sigmoid(self.perceptron(X))
                loss[i] = mean_squared_error(Y_pred, Y)
        if display_loss:
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squraed Error')
            plt.show()
            
    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.sigmoid(self.perceptron(x))
            Y_pred.append(y_pred)
        return Y_pred

"""def read_all(folder_path, key_prefix=""):
    '''
    It returns a dictionary with 'file names' as keys and 'flattened image arrays' as values.
    '''
    print("Reading:")
    images = {}
    files = os.listdir(folder_path)
    for i, file_name in tqdm_notebook(enumerate(files), total=len(files)):
        file_path = os.path.join(folder_path, file_name)
        image_index = key_prefix + file_name[:-4]
        image = Image.open(file_path)
        image = image.convert("L")
        images[image_index] = np.array(image.copy()).flatten()
        image.close()
    return images
"""

def read_all_my_train(folder_path):
    files = os.listdir(folder_path)
    Y = []
    X = np.array([], dtype=np.int32).reshape(0,256)
    
    for image in files:
        file_path = os.path.join(folder_path, image)
        image_data = Image.open(file_path)
        image_data = image_data.convert("L")
        image_data = np.array(image_data.copy()).flatten()
        image_data.reshape(1, 256)
        #np.concatenate((X, image_data))
        X = np.vstack([X, image_data]) if image_data.size else X
        #print(X)
        #break
        if image[0].isdigit():
            Y.append(0)
        else:
            Y.append(1)
    Y = np.asarray(Y)
    return X, Y
    
X_back, Y_back = read_all_my_train('C:/Users/hp/Desktop/GitHub projects/Deep Learning IIT/Deep Learning/Contest-2/train/background') 
Y_back = Y_back[:,np.newaxis]


X_en, Y_en = read_all_my_train('C:/Users/hp/Desktop/GitHub projects/Deep Learning IIT/Deep Learning/Contest-2/train/en')
Y_en = Y_en[:,np.newaxis]

X_back_en = np.concatenate((X_back, X_en))
Y_back_en = np.concatenate((Y_back, Y_en))

X_hi, Y_hi = read_all_my_train('C:/Users/hp/Desktop/GitHub projects/Deep Learning IIT/Deep Learning/Contest-2/train/hi')
Y_hi = Y_hi[:,np.newaxis]

X_back_en_hi = np.concatenate((X_back_en, X_hi))
Y_back_en_hi = np.concatenate((Y_back_en, Y_hi))

X_ta, Y_ta = read_all_my_train('C:/Users/hp/Desktop/GitHub projects/Deep Learning IIT/Deep Learning/Contest-2/train/ta')
Y_ta = Y_ta[:,np.newaxis]

X_back_en_hi_ta = np.concatenate((X_back_en_hi, X_ta))
Y_back_en_hi_ta = np.concatenate((Y_back_en_hi, Y_ta))

X = X_back_en_hi_ta
Y = Y_back_en_hi_ta

data = np.append(X, Y, axis=1)

# Shuffle
np.random.shuffle(data)

X = data[:, :-1]
Y = data[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0, test_size = 0.1)

# binarised
def binarised(x):
    return x > 0

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_binarised = sc_X.fit_transform(X_train)
X_test_binarised = sc_X.transform(X_test)

#X_train_binarised = binarised(X_train).astype(int)
#X_test_binarised = binarised(X_test).astype(int)

sigmoid = SigmoidNeuron()
sigmoid.fit(X_train_binarised, Y_train, epochs=200, learning_rate = 0.1)

Y_pred = sigmoid.predict(X_test_binarised)
Y_pred = np.array(Y_pred)

def bina(x):
    return x > 0.5

Y_pred = bina(Y_pred).astype(int)

result = confusion_matrix(Y_test,Y_pred)
print(result)
    
accuracy = accuracy_score(Y_test, Y_pred)
print(accuracy)
    



""" -------------------   Test  ------------------------------- """


def read_all_my_test(folder_path):
    files = os.listdir(folder_path)
    files = sorted(files,key=lambda x: int(os.path.splitext(x)[0]))
    X = np.array([], dtype=np.int32).reshape(0,256)
    
    for image in files:
        file_path = os.path.join(folder_path, image)
        image_data = Image.open(file_path)
        image_data = image_data.convert("L")
        image_data = np.array(image_data.copy()).flatten()
        image_data.reshape(1, 256)
        #np.concatenate((X, image_data))
        X = np.vstack([X, image_data]) if image_data.size else X
        #print(X)
        #break
    return X

def read_all_my_train(folder_path):
    files = os.listdir(folder_path)
    
    Y = []
    X = np.array([], dtype=np.int32).reshape(0,256)
    
    for image in files:
        file_path = os.path.join(folder_path, image)
        image_data = Image.open(file_path)
        image_data = image_data.convert("L")
        image_data = np.array(image_data.copy()).flatten()
        image_data.reshape(1, 256)
        #np.concatenate((X, image_data))
        X = np.vstack([X, image_data]) if image_data.size else X
        #print(X)
        #break
        if image[0].isdigit():
            Y.append(0)
        else:
            Y.append(1)
    Y = np.asarray(Y)
    return X, Y

""" 0 1 10 100 101 102 103 104 10five """

g_X = read_all_my_test('C:/Users/hp/Desktop/GitHub projects/Deep Learning IIT/Deep Learning/Contest-2/test')
    
g_X_test_binarised = sc_X.transform(g_X)

g_Y_pred = sigmoid.predict(g_X_test_binarised)
g_Y_pred = np.array(g_Y_pred)

g_Y_pred = bina(g_Y_pred).astype(int)

image_id = []
for i in range(300):
    image_id.append(i)
    
image_id = np.array(image_id)
image_id = image_id[:,np.newaxis]
image_id_list = list(image_id)

#submission['ImageId'] = image_id
#submission['Class'] = g_Y_pred

ans = []
for i in range(g_X.shape[0]):
    c = 0
    for j in range(g_X.shape[1]):
        if g_X[i, j] == 255:
            c = c + 1
    if c == 256:
        ans.append(0)
    else:
        ans.append(1)
g_Y_pred = np.array(ans)

csv_file = 'submission.csv'
sub_main_list = []

for key, value in zip(image_id, g_Y_pred):
    sub_li = []
    sub_li.append(key[0])
    sub_li.append(value)
    sub_main_list.append(sub_li)
    
sub_main_arr = np.array(sub_main_list)
    
submission_main_df = pd.DataFrame({'ImageId': sub_main_arr[:, 0], 'Class':sub_main_arr[:, 1]})
    
submission_main_df = submission_main_df[['ImageId', 'Class']]
submission_main_df = submission_main_df.sort_values(['ImageId'])
submission_main_df.to_csv(csv_file, index=False)



result = confusion_matrix(ans,g_Y_pred)
print(result)
    
accuracy = accuracy_score(ans, g_Y_pred)
print(accuracy)
            

    
    
    
    
    
    
    
    