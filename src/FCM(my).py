from math import sqrt, ceil
from random import gauss, random

import matplotlib.pyplot as plt
import pandas as pd

COLORS = ["gray", "blue", "green"]
DIM_NUM = 4
M_FACTOR = 1.1
CLUST_NUM = 3
ITER_NUM = 10

TABLE = {"Setosa": 0,
         "Versicolor":1,
         "Virginica" : 2}

def dataset(N): # DIM_NUM = 2
    res = []
    for i in range(N):
        p = random()
        
        if p < 0.33:
            clust = 0
            x = gauss(-2, 0.8)
            y = gauss(0, 0.4)
        elif p < 0.66:
            clust = 1
            x = gauss(-1, 0.8)
            y = gauss(1, 0.4)
        else:
            clust = 2
            x = gauss(0, 0.4)
            y = gauss(0.5, 0.8)

        res.append(point(clust, [x, y]))
    return res


class point:
    def __init__(self, clustNum, coords):
        self.clustNum = clustNum # Номер кластера, к которому принадлежит данная точка
        self.coords = coords

    def EuclideanDistance(self, other):
        res = 0
        for d in range(DIM_NUM):
            res += (self.coords[d] - other.coords[d]) * (self.coords[d] - other.coords[d])
        return sqrt(res)
    
    def evalCenter(self, PP, U):
        for d in range(DIM_NUM):
            termUp = 0
            termDown = 0
            for pi in range(len(PP)):
                termUp += (U[pi][self.clustNum] ** M_FACTOR) * PP[pi].coords[d]
                termDown += U[pi][self.clustNum] ** M_FACTOR
            self.coords[d] = termUp / termDown

def evalMrxU(CC, PP):
    res = []
    for pi in range(len(PP)):
        res.append([])
        for ci in range(len(CC)):
            tmp = 0
            for ck in range(len(CC)):
                tmp += (PP[pi].EuclideanDistance(CC[ci]) / PP[pi].EuclideanDistance(CC[ck])) ** (2/(M_FACTOR-1))
            res[pi].append(1/tmp)
    return res

# Data Initialization

CC = [point(i, [gauss(0, 4) for j in range(DIM_NUM)]) for i in range(CLUST_NUM)]

#PP = dataset(150)
PP = []
df = pd.read_csv("datasets/iris.csv")
for i in range(len(df)):
    coords = [df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3]]
    clust = TABLE[df.iloc[i, 4]]
    PP.append(point(clust, coords))

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
            ax.scatter(p.coords[i], p.coords[j], c=COLORS[p.clustNum], s=50)
        
        ax.set_xlabel('$Axis: ('+str(i)+', '+str(j)+')$', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.autoscale()
        n += 1
fig.tight_layout()
plt.show()

# Algorithm

currIter = 0

while currIter < ITER_NUM:
    U = evalMrxU(CC, PP)
    for pi, p in enumerate(PP):
        temp = max(U[pi])
        p.clustNum = U[pi].index(temp)
    
    currIter += 1
    for c in CC:
        c.evalCenter(PP, U)
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
            ax.scatter(p.coords[i], p.coords[j], c=COLORS[p.clustNum], s=50)
        
        ax.set_xlabel('$Axis: ('+str(i)+', '+str(j)+')$', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.autoscale()
        n += 1
fig.tight_layout()
plt.show()