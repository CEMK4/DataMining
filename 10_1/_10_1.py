import pandas as pd
from sklearn import preprocessing

weather = pd.read_csv('weather.csv').dropna()

weather.loc[weather['weather'] != "sun", 'weather'] = "rain"

ordinal_encoder = preprocessing.OrdinalEncoder(dtype=int)
data = pd.DataFrame(ordinal_encoder.fit_transform(weather), columns=weather.columns).drop('date', axis= 1)

print(data['weather'].value_counts())

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

Clust = [CLUST(data.values[0], data['weather'][1])]
i0 = 0
d0 = 0
for i in range(0, POINT_N):
    d = Clust[0].dist(data.values[i])
    if d > d0 and Clust[0].clust != data['weather'][i]:
        d0 = d
        i0 = i
Clust.append(CLUST(data.values[i0], data['weather'][i0]))
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
    
    r = Res['clust'] == data['weather']
    print('\nТочность = ', r.sum()/len(r))


