import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from math import exp
from math import sqrt

POINT_N = 300
DIM_N = 2
CLUST_N = 3

data, target = datasets.make_blobs(n_samples=POINT_N, centers=CLUST_N, n_features=DIM_N, center_box=(-10,10), random_state=0)
print('\nMulticluster')

A = 1.0
def sigma(x):
    return 1.0 / (1.0 + np.exp(-A*x))
def B(x, X):
    return exp(-0.10*np.dot((x-X),(x-X)))
def SoftMax(x):
    return np.exp(3.0*x)

class POINT:
    def __init__(self, clust, x, y):
        self.clust = clust
        self.x = x
        self.y = y

class CLUSTER(POINT):
    def __init__(self, clust, x, y):
        POINT.__init__(self, clust, x, y)
        self.n = 0

    def dist(self, p):
        return (self.x - p.x) ** 2 + (self.y - p.y) ** 2

    def eval(self, points):
        self.n = 0
        self.x = 0.0
        self.y = 0.0
        for i in range(POINT_N):
            if points[i] == self.clust:
                self.n += 1
                self.x += data[i][0]
                self.y += data[i][1]
        self.x /= self.n
        self.y /= self.n

class NNET:
    def __init__(self):
        self.input_nodes = DIM_N
        self.output_nodes = CLUST_N
        self.weights_input_to_output = 0.10 * np.random.randn(self.output_nodes, self.input_nodes)
        self.output_bias = np.zeros(self.output_nodes, dtype=float)
    def run(self, features):
        return sigma( np.dot(self.weights_input_to_output, features) + self.output_bias )

network = NNET()

trg = [[int(target[n]==i) for i in range(CLUST_N)] for n in range(POINT_N)]

u = np.empty(CLUST_N, dtype=float)
v = np.empty(CLUST_N, dtype=float)
p = np.empty(CLUST_N, dtype=float)

Color = ['blue', 'green', 'cyan', 'black']

CLUSTERS = [CLUSTER(0, 1, 1),CLUSTER(1, 1, -1),CLUSTER(2, -2, 0)]
POINTS = [CLUST_N for i in range(POINT_N)]

H = 0.01
eps = 0.003
for step in range(3):
    Center = [[clust.x, clust.y] for clust in CLUSTERS]    
    for iter in range(1000):
        sw = np.zeros([CLUST_N, DIM_N], dtype=float)
        sb = np.zeros(CLUST_N, dtype=float)
        for n in range(POINT_N):
            u = np.dot(network.weights_input_to_output, data[n]) + network.output_bias
            norm = sum(SoftMax(u))
            BS = 0.0
            for k in range(CLUST_N):
                BS += SoftMax(u[k])/norm * B(data[n], Center[k])
            for k in range(CLUST_N):
                p[k] = SoftMax(u[k])/norm * (B(data[n], Center[k]) - BS)
            sb += p
            for k in range(CLUST_N):
                for l in range(DIM_N):
                    sw[k][l] += p[k] * data[n][l]
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
        res = network.run(data[n])
        r0 = 0.0
        i0 = 0
        for i in range(CLUST_N):
            if res[i] > r0:
                i0 = i
                r0 = res[i]      
        POINTS[n] = i0

    
    print('eps=', eps, '     iter=', iter)
    print(1.0*sum(POINTS==target)/POINT_N)
    print(network.weights_input_to_output, network.output_bias, "\n")
    plt.figure(figsize=(8,8))
    for i in range(POINT_N):
        plt.scatter(data[i][0], data[i][1], c=Color[POINTS[i]], marker='o')
    for clust in CLUSTERS:
        plt.scatter(clust.x, clust.y, c='red', marker="*", s=100)
    plt.show()

    for c in CLUSTERS:
        c.eval(POINTS)
