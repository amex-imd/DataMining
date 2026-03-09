from random import random, gauss
from math import sin, cos, sqrt, ceil

import matplotlib.pyplot as plt
import pandas as pd

DIM_NUM = 2
EPSILON = 0.15
MIN_POINTS = 4
COLORS = ["blue", "green", "purple", "gray", "orange", "brown", "red", "magenta", 
          "cyan", "yellow", "pink", "lime", "teal", "lavender", "coral", "navy", 
          "gold", "indigo", "violet", "turquoise", "salmon", "plum", "olive", "maroon"]
TABLE = {"Setosa": 0,
         "Versicolor":1,
         "Virginica" : 2}

def dataIslands(N): # DIM_NUM = 2, EPSILON = 0.22, MIN_POINTS = 5
    res = []
    for _ in range(N):
        p = random()
        
        if p < 0.33:
            clust = 0
            x = gauss(-2, 0.3)
            y = gauss(0.5, 0.1)
        elif p < 0.66:
            clust = 1
            x = gauss(-2.4, 0.2)
            y = gauss(1, 0.2)
        else:
            clust = 2
            x = gauss(-1.6, 0.2)
            y = gauss(1.5, 0.3)

        res.append(point(clust, [x, y]))
    return res

def dataMoons(N): # DIM_NUM = 2, EPSILON = 1, MIN_POINTS = 5
    res = []
    for i in range(N):
        f = 3.14 * random()
        r = 0.2 * gauss(0.0, 0.4) + 0.9
        p = random()
        
        if p < 0.33:
            clustNum = 0
            x = 0.5 + r * cos(f)
            y = -0.25 + r * sin(f)
        elif p < 0.66:
            clustNum = 1
            x = -0.5 + r * cos(f)
            y = 0.25 - r * sin(f)
        else:
            clustNum = 2
            x = -1 + r * cos(f)
            y = -0.3 - r * sin(f)
        res.append(point(clustNum, [x, y]))
    return res


class point:
    def __init__(self, clustNum, coords):
        self.clustNum = clustNum # Номер кластера, к которому принадлежит данная точка
        self.coords = coords
        self.isVisited = False
        self.isNoise = False

    def EuclideanDistance(self, other):
        res = 0
        for d in range(DIM_NUM):
            res += (self.coords[d] - other.coords[d]) * (self.coords[d] - other.coords[d])
        return sqrt(res)
    
    def expand(self, PP, neighbours):
        ind = 0
        while ind < len(neighbours):
            curr = neighbours[ind]
            if not curr.isVisited:
                curr.isVisited = True
                tmp = curr.neighbours(PP)
                if len(tmp) >= MIN_POINTS:
                    for n in tmp:
                        if n not in neighbours:
                            neighbours.append(n)
            if curr.clustNum == -1:
                curr.clustNum = self.clustNum
            ind += 1

    def neighbours(self, PP):
        res = []
        for pi in range(len(PP)):
            if self.EuclideanDistance(PP[pi]) <= EPSILON:
                res.append(PP[pi])
        return res

def DBSCAN(PP):
    ci = 0
    for pi in range(len(PP)):
        if not PP[pi].isVisited:
            PP[pi].isVisited = True
            neighbours = PP[pi].neighbours(PP)
            if len(neighbours) < MIN_POINTS:
                PP[pi].isNoise = True
                PP[pi].clustNum = -1
            else:
                PP[pi].clustNum = ci
                PP[pi].expand(PP, neighbours)
                ci += 1

PP = dataIslands(150)
#PP = dataMoons(200)
"""PP = []
df = pd.read_csv("datasets/iris.csv")
for i in range(len(df)):
    coords = [df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3]]
    clust = TABLE[df.iloc[i, 4]]
    PP.append(point(clust, coords))
# EPSILON = 0.6, MIN_POINT = 5"""

plotsNum = DIM_NUM * (DIM_NUM - 1) // 2
colsNum = min(3, plotsNum)
rowsNum = ceil(plotsNum / colsNum)

fig, axes = plt.subplots(rowsNum, colsNum, figsize=(10, 6))

if rowsNum == 1 and colsNum == 1:
    axes = [[axes]]
elif rowsNum == 1:
    axes = [axes]
elif colsNum == 1:
    axes = [[ax] for ax in axes]

n = 0
for i in range(DIM_NUM):
    for j in range(i+1, DIM_NUM):
        if rowsNum == 1 and colsNum == 1:
            ax = axes[0][0]
        elif rowsNum == 1:
            ax = axes[n]
        elif colsNum == 1:
            ax = axes[n][0]
        else:
            ax = axes[n // colsNum][n % colsNum]
        
        for p in PP:
            if p.isNoise:
                ax.scatter(p.coords[i], p.coords[j], c='black', s=20, marker='x')
            else:
                ax.scatter(p.coords[i], p.coords[j], c=COLORS[p.clustNum], s=50)
        
        ax.set_xlabel('$Axis: ('+str(i)+', '+str(j)+')$', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.autoscale()
        n += 1
fig.tight_layout()
plt.show()

for p in PP:
    p.clustNum = -1
    p.isVisited = False
    p.isNoise = False

DBSCAN(PP)

fig, axes = plt.subplots(rowsNum, colsNum, figsize=(10, 6))
if rowsNum == 1 and colsNum == 1:
    axes = [[axes]]
elif rowsNum == 1:
    axes = [axes]
elif colsNum == 1:
    axes = [[ax] for ax in axes]

n = 0
for i in range(DIM_NUM):
    for j in range(i+1, DIM_NUM):
        if rowsNum == 1 and colsNum == 1:
            ax = axes[0][0]
        elif rowsNum == 1:
            ax = axes[n]
        elif colsNum == 1:
            ax = axes[n][0]
        else:
            ax = axes[n // colsNum][n % colsNum]
        
        for p in PP:
            if p.isNoise:
                ax.scatter(p.coords[i], p.coords[j], c='black', s=20, marker='x')
            else:
                ax.scatter(p.coords[i], p.coords[j], c=COLORS[p.clustNum], s=50)
        
        ax.set_xlabel('$Axis: ('+str(i)+', '+str(j)+')$', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.autoscale()
        n += 1

fig.tight_layout()
plt.show()