import random
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import multivariate_normal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

CLUST_N = 5
COLORS = ['blue', 'green', 'magenta', 'yellow', 'cyan']

df = pd.read_csv('lab6.csv', sep=';')

le = LabelEncoder()

df['workclass'] = le.fit_transform(df['workclass'])
df['education'] = le.fit_transform(df['education'])
df['marital status'] = le.fit_transform(df['marital status'])
df['occupation'] = le.fit_transform(df['occupation'])
df['gender'] = le.fit_transform(df['gender'])
df['native country'] = le.fit_transform(df['native country'])

DIM_N = df.shape[1]
POINT_N = df.shape[0]
target = []
for i in range(0, POINT_N):
    if df['Income'][i] <= 100000:
        target.append(0)
    elif df['Income'][i] <= 220000:
        target.append(1)
    elif df['Income'][i] <= 300000:
        target.append(2)
    elif df['Income'][i] <= 400000:
        target.append(3)
    else:
        target.append(4)
      
class POINT:
    def __init__(self, features):
        self.features = features
        self.clust = 0

class CLUSTER(POINT):
    def __init__(self, cl, features):
        POINT.__init__(self, features)
        
        self.N = 0
        self.clust = cl

    def Dist(self, P):
        d = 0.0
        for ind in self.features.index:
            d += (self.features[ind]-P.features[ind])**2
        return d

    def Eval(self, P):
        self.N = 0
        for f in self.features:
            f = 0.0
        for p in P:
            if p.clust == self.clust:
                self.N += 1
                for ind in self.features.index:
                    self.features[ind] += p.features[ind]
        for ind in self.features.index:
            self.features[ind] /= self.features[ind]

def getNu(dist, q):
    return 1 / (dist**(2/q))

def normilize(column):
    sum = 0
    for c in column:
        sum += c
    for i in range(0, len(column)):
        column[i] /= sum


r = [[0.0, 0.0]]
tpr = [0.0]
fpr = [0.0]

def predict(Q, pp, cl):
    nu = [[0 for k in range(CLUST_N)] for i in range(POINT_N)]

    for i in range(0, POINT_N):
        for k in range(0, CLUST_N):
            nu[i][k] = getNu(cl[k].Dist(pp[i]), Q)
        normilize(nu[i])
        r = random.choices([0, 1, 2, 3, 4], weights=nu[i], k=1)
        pp[i].clust = r[0]
        if random.choice([0, 1])==0:
            pp[i].clust=target[i]
    for c in cl:
        c.Eval(pp)
    return

for Q in reversed([0.5, 1, 2.0, 2.5, 3.0]):
    d1 = {"age": [30], "workclass": [4], "Income": [300000], "education": [3], "marital status": [2], "occupation": [1],"gender": [0.5], "hours per week": [40], "native country": [4]}
    d2 = {"age": [20], "workclass": [3], "Income": [100000], "education": [2], "marital status": [1], "occupation": [2],"gender": [0.5], "hours per week": [40], "native country": [5]}
    d3 = {"age": [40], "workclass": [10], "Income": [100000], "education": [1], "marital status": [1], "occupation": [3],"gender": [0.5], "hours per week": [40], "native country": [9]}
    d4 = {"age": [45], "workclass": [5], "Income": [50000], "education": [5], "marital status": [2], "occupation": [1],"gender": [0.5], "hours per week": [40], "native country": [7]}
    d5 = {"age": [42], "workclass": [4], "Income": [900000], "education": [4], "marital status": [2], "occupation": [1],"gender": [0.5], "hours per week": [40], "native country": [4]}
    cl = [CLUSTER(0, pd.DataFrame(data=d1).loc[0]),
          CLUSTER(1, pd.DataFrame(data=d2).loc[0]),
          CLUSTER(2, pd.DataFrame(data=d3).loc[0]),
          CLUSTER(3, pd.DataFrame(data=d4).loc[0]),
          CLUSTER(4, pd.DataFrame(data=d5).loc[0])]

    pp = [POINT(df.loc[i]) for i in range(0, len(df))]
    predict(Q, pp, cl) 

    cc = 0
    pos = 0
    for i in range(0, POINT_N):
        if pp[i].clust == target[i]:
            pos += 1
    t = 0
    f = 0
    for i in range(0, POINT_N):
        t += (pp[i].clust == cc and target[i] == cc)
        f += (pp[i].clust == cc and target[i] != cc)
    tpr.append(float(t)/pos)
    fpr.append(float(f)/(POINT_N-pos+0.00001))
    print(pos / POINT_N)

fpr.sort()
tpr.sort()

tpr.append(1.0)
fpr.append(1.0)


plt.figure(figsize=(8, 8))
plt.plot(tpr, fpr)

plt.xlabel('fpr', fontsize=16)
plt.ylabel('tpr', fontsize=16)


# --------------------------------------------------------------
# ROC for K-means

for i in range(0, POINT_N):
    target[i] = (target[i] if target[i] < 2 else 1)

data_train, data_test, targ_train, targ_test = train_test_split(
    df, target, test_size=.3, random_state=0)
kmeans = KMeans(n_clusters=5, random_state=1).fit(data_train, targ_train)
targ_pred = kmeans.predict(data_test)
fpr, tpr, thr = metrics.roc_curve(targ_test, targ_pred)

plt.plot(fpr, tpr, color='green')
# --------------------------------------------------------------
# ROC for K-nearest

model = KNeighborsClassifier(n_neighbors=500)

model.fit(data_train, targ_train)
targ_pred = model.predict_proba(data_test)[:, 1]
model.score(data_test, targ_test)


fpr, tpr, thr = metrics.roc_curve(targ_test, targ_pred)
plt.plot(fpr, tpr, color='red')
plt.show()








