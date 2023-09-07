import pandas as pd
from sklearn import preprocessing

salary = pd.read_csv('salary.csv').dropna()

ordinal_encoder = preprocessing.OrdinalEncoder(dtype=int)
salary = pd.DataFrame(ordinal_encoder.fit_transform(salary), columns=salary.columns)

m = salary['Age'].mean()
s = salary['Age'].std()
salary['Age'] = (salary['Age'] - m)/s

m = salary['Salary'].mean()
s = salary['Salary'].std()
salary['Salary'] = (salary['Salary'] - m)/s

data = salary.drop('Gender', axis = 1)

POINT_N = len(data)
DIM_N = len(data.values[0])
CLUST_N = 2

class CLUST:
    def __init__(self, x, c):
        self.X = x
        self.clust = c
    def dist(self, X):
        d = (self.X != X).sum()
        return d
    def eval(self, df):
        cols = df.columns
        self.X = []
        for c in cols:
            self.X.append( df[c].value_counts().idxmax())
        return

Clust = [CLUST(data.values[0], salary['Gender'][1])]
i0 = 0
d0 = 0
for i in range(0, POINT_N):
    d = Clust[0].dist(data.values[i])
    if d > d0 and Clust[0].clust != salary['Gender'][i]:
        d0 = d
        i0 = i
Clust.append(CLUST(data.values[i0], salary['Gender'][i0]))
i0 = 0
d0 = 0

print(Clust[0].X)
print(Clust[0].clust)
print(Clust[1].X)
print(Clust[1].clust)

Res = pd.DataFrame(data=[CLUST_N for i in range(0, POINT_N)], columns=['clust'])

for n in range(0,3):
    for i in range(0, POINT_N):
        if Clust[0].dist(data.values[i]) < Clust[1].dist(data.values[i]):
            Res['clust'][i] = Clust[0].clust
        else:
            Res['clust'][i] = Clust[1].clust
    for cl in Clust:
        df = data.loc[ Res['clust'] == cl.clust]
        cl.eval(df)
        print('\n', cl.X)
    
    r = Res['clust'] == salary['Gender']
    print('\nТочность = ', r.sum()/len(r))
