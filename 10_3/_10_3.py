
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from math import exp
from math import sqrt

water = pd.read_csv('water.csv').dropna().head(400)

print (water['Potability'].value_counts())

target = water['Potability'].values
water = water.drop('Potability', axis=1)

for c in water.columns:
    m = water[c].mean()
    s = water[c].std()
    water[c] = (water[c] - m)/s

d0 = water.loc[target == 0]
d1 = water.loc[target == 1]

c0 = d0.iloc[0]
c1 = d1.iloc[0]

for c in d0.columns:
    c0[c] = d0[c].mean() 
    c1[c] = d1[c].mean()

Center = [c0.values, c1.values] 

data = water.values

POINT_N = len(data)
DIM_N = len(data[0])
CLUST_N = 2
DEFALT_LAYER = 2
NEURONS = [[5, CLUST_N],[5, 5, CLUST_N],[5, 5, 5, CLUST_N]]

A = 1.0
def Sigma(x):
    return 1.0 / (1.0 + np.exp(-A*x))
def SoftMax(x):
    x_max = max(x)
    x -= x_max
    return np.exp(3.0*x)/sum(np.exp(3.0*x))
def Near(x, X):
    return exp(-0.10*np.dot((x-X),(x-X)))

np.random.seed(1)
class LAYER:
    def __init__(self, n_neurons, n_input):
        self.n_neurons = n_neurons
        self.n_input = n_input
        self.weights = 0.1 * np.random.randn(self.n_neurons, self.n_input)
        self.bias = np.zeros(self.n_neurons, dtype=float)
    def forward(self, inputs):
        return np.dot(self.weights, inputs) + self.bias

for step in range(3):
    NEURON_N = NEURONS[step] 
    LAYER_N = DEFALT_LAYER + step
    
    Layer = [LAYER(NEURON_N[0], DIM_N)]
    for l in range(1, LAYER_N):
        Layer.append(LAYER(NEURON_N[l], NEURON_N[l-1]))

    H = 0.01
    for iter in range(300):
        sw = [np.zeros([NEURON_N[0], DIM_N], dtype=float)]
        sb = [np.zeros(NEURON_N[0], dtype=float)]
        for l in range(1, LAYER_N):
            sw.append(np.zeros([NEURON_N[l], NEURON_N[l-1]], dtype=float))
            sb.append(np.zeros(NEURON_N[l], dtype=float))
        x = np.empty(LAYER_N+1, dtype=object)       # Layer's input
        u = np.empty(LAYER_N, dtype=object)         # Layer's output
        e = np.empty(LAYER_N, dtype=object)
        for n in range(POINT_N):
            x[0] = data[n]
            for l in range(LAYER_N):
                u[l] = Layer[l].forward(x[l])
                x[l+1] = Sigma(u[l])

            SN = 0.0
            for k in range(CLUST_N):
                SN += SoftMax(u[LAYER_N-1])[k] * Near(data[n], Center[k])

            e[LAYER_N-1] = np.empty(len(u[LAYER_N-1]), dtype=float)
            for k in range(CLUST_N):
                e[LAYER_N-1][k] = SoftMax(u[LAYER_N-1])[k] * (Near(data[n],Center[k]) - SN)
            for l in range(LAYER_N-2, -1, -1):
                e[l] = np.empty(len(u[l]), dtype=float)
                for k in range(len(u[l])):
                    e[l][k] = np.dot(Layer[l+1].weights.T, e[l+1])[k] * Sigma(u[l][k]) * (1.0 - Sigma(u[l][k]))

            for l in range(LAYER_N):
                sb[l] += e[l]
            for l in range(0, LAYER_N):
                for k in range(len(e[l])):
                    for i in range(len(x[l])):
                        sw[l][k][i] += e[l][k] * x[l][i]

        for l in range(LAYER_N):
            Layer[l].bias += H * sb[l]
            Layer[l].weights += H * sw[l]

    pred = []
    for n in range(POINT_N):
        res = data[n]
        for l in range(LAYER_N):
            res = Layer[l].forward(res)
        r0 = 0.0
        i0 = 0
        for i in range(CLUST_N):
            if res[i] > r0:
                i0 = i
                r0 = res[i]
        pred.append(i0)   


    print('Layers =', LAYER_N)
    print('accuracy =' ,1.0*sum(pred==target)/POINT_N, '\n')
    

"""
Layers = 2
accuracy = 0.61

Layers = 3
accuracy = 0.745

Layers = 4
accuracy = 0.745

С ростом числа слоёв наблюдается повышение точности кластеризации данных. 
На данном примере входных данных максимальная точность была достигнута уже на 3-ёх слойной нейросети
"""

