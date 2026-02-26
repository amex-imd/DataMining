
import matplotlib.pyplot as plt
#from sklearn.datasets import make_blobs
from sklearn import datasets

import random
from math import sqrt
from math import pow

DIM_N = 2
POINT_N = 300
CLUST_N = 2

class POINT:
    def __init__(self, cl, x, y):
        self.clust = cl
        self.X = x
        self.Y = y

class CLUSTER(POINT):
    def __init__(self, cl, x, y):
        POINT.__init__(self, cl, x, y)
        self.N = 0
    def Dist(self, p):
        return sqrt( (self.X - p.X)**2 + (self.Y - p.Y)**2 )
    def Eval_Center(self, P, M):
        self.N = 0
        self.X = 0.0
        self.Y = 0.0
        a = 0.0
        for i in range(POINT_N):
            if P[i].clust == self.clust:
                self.N += 1
                self.X += P[i].X*M[i][self.clust]
                self.Y += P[i].Y*M[i][self.clust]
                a = a + M[i][self.clust]
        self.X /= a
        self.Y /= a

centers = [(0, 0), (2, 2)]
data, targ = datasets.make_blobs(n_samples=POINT_N, centers=centers, shuffle=False, random_state=42)

for i in range(POINT_N):
        if targ[i] == 0:
            plt.scatter(data[i][0], data[i][1], c='blue', s=20)
        else:
            plt.scatter(data[i][0], data[i][1], c='green', s=20)
plt.show()

Cl = [CLUSTER(0, 2.0*random.random()-1.0, 2.0*random.random()-1.0), CLUSTER(1, 2.0*random.random()-1.0, 2.0*random.random()-1.0)]
#Cl = [CLUSTER(0, 0, 0), CLUSTER(1, 2, 2)]

PP = [POINT(CLUST_N, data[i,0], data[i,1]) for i in range(POINT_N)]

QU = 0.1

for nn in range(10):
    CC = [POINT(0, Cl[0].X, Cl[0].Y), POINT(1, Cl[1].X, Cl[1].Y)]

    Mu = []
    for i in range(POINT_N):
        Mu.append([])
        a = 0.0
        for k in range(CLUST_N):
            Mu[i].append( pow(Cl[k].Dist(PP[i]), 1.0/QU) )
            a = a + Mu[i][k]
        for k in range(CLUST_N):
            Mu[i][k] = Mu[i][k]/a

    for i in range(POINT_N):
        r = random.random()
        if r < Mu[i][0]:
            PP[i].clust = 0
        else:
            PP[i].clust = 1

    for k in range(CLUST_N):
        Cl[k].Eval_Center(PP, Mu)

    for p in PP:
        if p.clust == 0:
            plt.scatter(p.X, p.Y, c='blue', s=20)
        else:
            plt.scatter(p.X, p.Y, c='green', s=20)
    for cl in Cl:
            plt.scatter(CC[cl.clust].X, CC[cl.clust].Y, c='cyan', marker="*")
            plt.scatter(cl.X, cl.Y, c='red', marker="*")
    plt.show()


