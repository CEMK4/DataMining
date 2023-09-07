import matplotlib.pyplot as plt
import random
import math

CLUST_NUM = 2
POINT_NUM = 300
KN_NUM = 5 


class Point:
    
    def __init__(self, x, y):
        self.X = x
        self.Y = y
        self.Cluster = CLUST_NUM
        self.Kn = [0 for i in range(0, KN_NUM)]
        self.dens = 0.0

    def Neighbours(self, A):
        self.dens = 0.0
        l = [i for i in range(0, len(A))]
        for k in range(0, KN_NUM):
            n = -1
            d = 1.0e+99
            for i in l:
                if d > (self.X-A[i].X)**2 + (self.Y-A[i].Y)**2:
                    n = i
                    d = (self.X-A[i].X)**2 + (self.Y-A[i].Y)**2
            self.dens += d
            self.Kn[k] = n
            l.remove(n)
        self.dens /= KN_NUM

def DataClouds(A, N):
    for i in range(0, N):
        if random.random() < 0.5:
            A[i].clust = 0
            A[i].X = random.normalvariate(-0.75, 0.5)
            A[i].Y = random.normalvariate(0.0, 0.5)
        else:
            A[i].clust = 1
            A[i].X = random.normalvariate(0.0, 0.5)
            A[i].Y = random.normalvariate(0.0, 0.5)
    return

def DataMoons(A,N):
    for i in range (0,N):
        deg = 3.14 * random.random()
        r = 0.2 * random.normalvariate(0.0, 0.4) + 0.9
        if random.random() < 0.5:
            A[i].X = 0.5 + r*math.cos(deg)
            A[i].Y = -0.25 + r*math.sin(deg)
            A[i].Cluster = 0
        else:
            A[i].X = -0.5 + r*math.cos(deg)
            A[i].Y = 0.25 - r*math.sin(deg)
            A[i].Cluster = 1
    return

def RecurClust(p, cl, PP):
    neigh = PP[p.Kn[CLUST_NUM - 1]]
    while True:
        if neigh.Cluster == CLUST_NUM:
            if neigh.dens <= PP[neigh.Kn[CLUST_NUM - 1]].dens:
                for i in range(0,CLUST_NUM):
                    if cl[i] == CLUST_NUM:
                        num = i
                        break
                neigh.Cluster = num
                break
            else:
                RecurClust(neigh, cl)
        else:
            p.Cluster = neigh.Cluster
            break
    p.Cluster = neigh.Cluster


PP = [Point(0.0,0.0) for i in range(0, POINT_NUM)]
DataClouds(PP, POINT_NUM)

cl = [CLUST_NUM for i in range (0, CLUST_NUM)]

for p in PP:
    p.Neighbours(PP)
    RecurClust(p, cl, PP)
    if p.Cluster == 0:
        plt.scatter(p.X, p.Y, c='green', marker='.')
    else:
        plt.scatter(p.X, p.Y, c='blue', marker='.')
    

plt.show()

PP2 = [Point(0.0,0.0) for i in range(0, POINT_NUM)]
DataMoons(PP2, POINT_NUM)

cl2 = [CLUST_NUM for i in range (0, CLUST_NUM)]
for p in PP2:
    p.Neighbours(PP2)
    RecurClust(p, cl2, PP2)
    if p.Cluster == 0:
        plt.scatter(p.X, p.Y, c='green', marker='.')
    else:
        plt.scatter(p.X, p.Y, c='blue', marker='.')
    

plt.show()
