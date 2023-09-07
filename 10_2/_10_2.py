import pandas as pd
from sklearn import preprocessing
import numpy as np
from math import exp
from collections import Counter

mush = pd.read_csv('mushrooms.csv')
ordinal_encoder = preprocessing.OrdinalEncoder(dtype=int)
mush = pd.DataFrame( ordinal_encoder.fit_transform(mush), columns=mush.columns )

data = mush.drop('class', axis=1)

POINT_N = len(data)
DIM_N = 22
CLUST_N = 2

A = 1.0
def sigma(x):
    return 1.0 / (1.0 + np.exp(-A*x))
def delta(x):
    return np.exp(-A*np.square(x*x))

np.random.seed(5)

class NNET4:
    def __init__(self):
        self.input_nodes = DIM_N
        self.output_nodes = CLUST_N
        self.weights_input_to_output = np.random.rand(self.output_nodes, self.input_nodes)
        self.output_bias = np.zeros(self.output_nodes, dtype=float)
    def run(self, features):
        return sigma( np.dot(self.weights_input_to_output, features) + self.output_bias )

u = np.empty(CLUST_N, dtype=float)
v = np.empty(CLUST_N, dtype=float)
p = np.empty(CLUST_N, dtype=float)

H = 0.1
eps = 0.001

network = NNET4()

for iter in range(300):
    sw = np.zeros([CLUST_N, DIM_N], dtype=float)
    sb = np.zeros(CLUST_N, dtype=float)
    for n in range(POINT_N):
        u = np.dot(network.weights_input_to_output, data.values[n]) + network.output_bias
        v = mush['class'][n] - sigma(u)
        p = v*delta(v)*sigma(u)*(1.0-sigma(u))
        sb += p
        for k in range(CLUST_N):
            for l in range(DIM_N):
                sw[k][l] += p[k] * data.values[n][l]
    network.output_bias += H * sb
    network.weights_input_to_output += H * sw
    dw2 = 0.0
    db2 = 0.0
    for i in range(network.output_nodes):
        for j in range(network.input_nodes):
            dw2 += sw[i][j]*sw[i][j]
        db2 += sb[i]*sb[i]
    if dw2*H**2 < eps**2 and db2*H**2 < eps**2:
        break

pred = []
for n in range(POINT_N):
    res = network.run(data.values[n])
    r0 = 0.0
    i0 = 0
    for i in range(CLUST_N):
        if res[i] > r0:
            i0 = i
            r0 = res[i]
    pred.append(i0)

accuracy = 1.0*sum(pred==mush['class'])/POINT_N

print('eps=', eps, '     iter=', iter)
print('точность =', accuracy)

