import matplotlib.pyplot as plt
import random

CLUST_N = 2
POINT_N = 300


def DataClouds(points, n):
    for i in range(0, n):
        if random.random() < 0.5:
            points[i].clust = 0
            points[i].x = random.normalvariate(-0.25, 0.75)
            points[i].y = random.normalvariate(0.0, 0.2)
        else:
            points[i].clust = 1
            points[i].x = random.normalvariate(-0.25, 0.75)
            points[i].y = random.normalvariate(0.0, 0.8)
    return

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
        for p in points:
            if p.clust == self.clust:
                self.n += 1
                self.x += p.x
                self.y += p.y
        self.x /= self.n
        self.y /= self.n


CLUSTERS = [CLUSTER(0, -2.0, 0.5), CLUSTER(1, 2.0, 0.5)]
POINTS = [POINT(CLUST_N, 0.0, 0.0) for i in range(0, POINT_N)]
DataClouds(POINTS, POINT_N)

for point in POINTS:
    if point.clust == 0:
        plt.scatter(point.x, point.y, c='green', s=20)
    elif point.clust == 1:
        plt.scatter(point.x, point.y, c='blue', s=20)
plt.show()

Q = 0.25
PROBABILITIES = [[0 for i in range(0, POINT_N)] for i in range(0, CLUST_N)]

for n in range(0,3):
    for cn in range(0, CLUST_N):
        for pn in range(0, POINT_N):
            PROBABILITIES[cn][pn] = 1 / (CLUSTERS[cn].dist(POINTS[pn]) ** (1/Q))

    for pn in range(0, POINT_N):
        sumProbabilities = 0
        for cn in range(0, CLUST_N):
            sumProbabilities += PROBABILITIES[cn][pn]
        A = 1 / sumProbabilities
        for cn in range(0, CLUST_N):
            PROBABILITIES[cn][pn] *= A

    for pn in range(0, POINT_N):
        point = POINTS[pn]
        if random.random() < PROBABILITIES[0][pn]:
            point.clust = 0
            plt.scatter(point.x, point.y, c='green', s=20)
        else:
            point.clust = 1
            plt.scatter(point.x, point.y, c='blue', s=20)

    for c in CLUSTERS:
        plt.scatter(c.x, c.y, c='red', marker='*', s=30)
    plt.show()

    for c in CLUSTERS:
        c.eval(POINTS)
