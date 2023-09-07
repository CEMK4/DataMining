import pandas as pd
import matplotlib.pyplot as plt

bills = pd.read_csv("fake_bills.csv")
cat_columns = bills.select_dtypes(['bool']).columns
bills[cat_columns] = bills[cat_columns].apply(lambda x: pd.factorize(x)[0])

is_genuine = bills['is_genuine'].values

for i in range(0, len(bills)):
    plt.scatter(bills['height_left'][i], bills['height_right'][i], color = (1.0-is_genuine[i], 0.0, is_genuine[i]), s=20)
plt.show()

KNBR_N = 5

class POINT:
    def __init__(self, x, y, diagonal, length):
        self.x = x
        self.y = y
        self.diagonal = diagonal
        self.length = length
        self.kn = [0 for i in range(0, KNBR_N)]
        self.clust = 0

    def neighbors(self, A):
        l = [i for i in range(0, len(A))]
        cl = [0,0]
        for k in range(0, KNBR_N):
            i0 = 0
            d0 = 1.0e+99
            cl[A['is_genuine'][i0]] += 1
            for i in l:
                d = (self.x - A['height_left'][i])**2 + (self.y - A['height_right'][i])**2 # 69%
                d += (self.diagonal - A['diagonal'][i])**2                                 # 74%
                d += (self.length - A['length'][i])**2                                     # 96%

                if d0 > d and d > 0.0:
                    cl[A['is_genuine'][i0]] -= 1
                    i0 = i
                    cl[A['is_genuine'][i0]] += 1
                    d0 = d
            self.kn[k] = i0
            l.remove(i0)
        if cl[0] > cl[1]:
            self.clust = 0
        else:
            self.clust = 1


PP = [POINT(bills['height_left'][i], bills['height_right'][i], bills['diagonal'][i], bills['length'][i]) for i in range(0, len(bills))]

for p in PP:
    p.neighbors(bills)
    plt.scatter(p.x, p.y, color=(1.0-float(p.clust), 0.0, float(p.clust)), s=20)
plt.show()

n = 0
for i in range(0, len(bills)):
    if PP[i].clust == bills['is_genuine'][i]:
        n += 1

print('=====>', float(n)/len(bills))
