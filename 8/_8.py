
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from math import exp

"""
data = pd.read_csv('dataset2.csv').dropna()
#data.loc[data['style'] != "IPA", 'style'] = "Lager"
ordinal_encoder = preprocessing.OrdinalEncoder(dtype=int)
data = pd.DataFrame(ordinal_encoder.fit_transform(data), columns=data.columns).drop('Brew No.', axis=1)

for c in data.columns:
    m = data[c].mean()
    s = data[c].std()
    data[c] = (data[c] - m)/s

target = data['style'].values
data = data.drop(['style'], axis= 1).values

POINT_N = len(data)
DIM_N = len(data[0])
CLUST_N = 3

A = 1.0
def sigma(x):
    return 1.0 / (1.0 + np.exp(-A*x))
def delta(x):
    return np.exp(-A*np.square(x*x))

np.random.seed(3)
class NNET3:
    def __init__(self):
        self.input_nodes = DIM_N
        self.weights_input_to_output = np.random.rand(self.input_nodes)
        self.output_bias = 0.0
    def activation_function(self, x):
        return sigma(x)
    def run(self, features):
        input_output = np.dot(features, self.weights_input_to_output)
        return self.activation_function(input_output+self.output_bias)

network = NNET3()

H = 0.1
eps = 0.01
for i in range(300):
    sw = np.zeros(DIM_N, dtype=float)
    sb = 0.0
    for n in range(POINT_N):
        u = np.dot(data[n], network.weights_input_to_output) + network.output_bias
        v = target[n] - sigma(u)
        t = v*delta(v)*sigma(u)*(1.0-sigma(u))
        sb += t
        sw += t*data[n]
    network.output_bias += H * sb
    network.weights_input_to_output += H * sw
    if sb*H < eps and np.dot(sw, sw)*H**2 < eps**2:
        break
print('eps=', eps, '     iter=', i)

pred = []
for n in range(POINT_N):
    res = network.run(data[n])
    #pred.append((1 if res>0.5 else 0))
    if res > 2/3:
        pred.append(2)
    elif res > 1/3:
        pred.append(1)
    else:
        pred.append(0)

print(1.0*sum(pred==target)/POINT_N)
print(network.weights_input_to_output, network.output_bias)
"""

data = pd.read_csv('weather.csv').dropna()
data.loc[data['weather'] != "sun", 'weather'] = "rain"


ordinal_encoder = preprocessing.OrdinalEncoder(dtype=int)
data = pd.DataFrame(ordinal_encoder.fit_transform(data), columns=data.columns).drop(['date','temp_min','temp_max'], axis= 1)


target = data['weather'].values
data = data.drop('weather', axis= 1).values


POINT_N = len(data)
DIM_N = len(data[0])
CLUST_N = 2


A = 1.0
def sigma(x):
    return 1.0 / (1.0 + np.exp(-A*x))
def delta(x):
    return np.exp(-A*np.square(x*x))

np.random.seed(6)
class NNET3:
    def __init__(self):
        self.input_nodes = DIM_N
        self.weights_input_to_output = np.random.rand(self.input_nodes)
        self.output_bias = 0.0
    def activation_function(self, x):
        return sigma(x)
    def run(self, features):
        input_output = np.dot(features, self.weights_input_to_output)
        return self.activation_function(input_output+self.output_bias)

network = NNET3()

H = 0.1
eps = 0.01
for i in range(300):
    sw = np.zeros(DIM_N, dtype=float)
    sb = 0.0
    for n in range(POINT_N):
        u = np.dot(data[n], network.weights_input_to_output) + network.output_bias
        v = target[n] - sigma(u)
        t = v*delta(v)*sigma(u)*(1.0-sigma(u))
        sb += t
        sw += t*data[n]
    network.output_bias += H * sb
    network.weights_input_to_output += H * sw
    if sb*H < eps and np.dot(sw, sw)*H**2 < eps**2:
        break
print('eps=', eps, '     iter=', i)

pred = []
for n in range(POINT_N):
    res = network.run(data[n])
    pred.append((1 if res>0.5 else 0))

print(1.0*sum(pred==target)/POINT_N)
print(network.weights_input_to_output, network.output_bias)
