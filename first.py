import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy import linalg
import psutil

train = pd.read_csv('MNIST_data_train.csv')
test = pd.read_csv('MNIST_data_test.csv')


df = train.head
df


onehotencoder = OneHotEncoder(choices='auto')
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train.values)
y_predicted_train = pd.read_csv('MNIST_target_train.csv')
y_train = onehotencoder.fit_transform(y_predicted_train.values[:,:1]).toarray()

input_size = X_train.shape[1]


hidden_size = 1000

input_weights = np.random.normal(size=[input_size,hidden_size])
biases = np.random.normal(size=[hidden_size])

def relu(x):
   return np.maximum(x, 0, x)



def hidden_nodes(X):
    G = np.dot(X, input_weights)
    G = G + biases
    H = relu(G)
    return H


psutil.cpu_times()
output_weights = np.dot(scipy.linalg.pinv(hidden_nodes(X_train)), y_train)

def predict(X):
    out = hidden_nodes(X)
    out = np.dot(out, output_weights)
    return out


a = predict(X_train)

onehotencoder = OneHotEncoder(categories='auto')
scaler = MinMaxScaler()
X_test = scaler.fit_transform(test.values)
y_test_predicted = pd.read_csv('MNIST_target_test.csv')
y_test = onehotencoder.fit_transform(y_test_predicted.values[:,:1]).toarray()
# X_test = scaler.fit_transform(test.values)
# y_test = pd.read_csv('MNIST_target.csv')

start_time = time.time()
f = predict(X_test)
end_time = time.time()
dur_skl=end_time-start_time
print(dur_skl)
correct = 0
total = X_test.shape[0]
cpu_stats = psutil.virtual_memory()
print(cpu_stats)
for i in range(total):
    predicted = np.argmax(f[i])
    actual = np.argmax(y_test[i])
    correct += 1 if predicted == actual else 0  
accuracy = correct/total
print('Accuracy for ', hidden_size, ' hidden nodes: ', accuracy)

print(f)

correct = 0
total = a.shape[0]
for i in range(total):
    predicted = np.argmax(a[i])
    actual = np.argmax(y_train[i])
    correct += 1 if predicted == actual else 0  
accuracy_train = correct/total

print(accuracy_train)