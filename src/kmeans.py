import numpy as np
from math import sqrt, sin, cos
from random import random, normalvariate
import matplotlib.pyplot as plt
from functools import reduce


POINTS_NUM: int = 300
CLUST_NUM: int = 3
EPSILON: float = 0.05


def dataClouds():
    res = np.array([point2D(0, 0, i) for i in range(POINTS_NUM)])
    for i in range(POINTS_NUM):
        p = random()
        if p < 0.33: 
            res[i].clustNum = 0
            res[i].x = normalvariate(-1.0, 0.1)
            res[i].y = normalvariate(0.0, 0.7)
        elif p < 0.66:
            res[i].clustNum = 1
            res[i].x = normalvariate(1.0, 0.2)
            res[i].y = normalvariate(0.0, 0.3)
        else:
            res[i].clustNum = 2
            res[i].x = normalvariate(0.0, 0.2)
            res[i].y = normalvariate(-1.0, 0.1)
    return res

def dataMoon():
    res = np.array([point2D(0, 0, i) for i in range(POINTS_NUM)])
    for i in range(POINTS_NUM):
        f = 3.14 * random()
        r = 0.2 * normalvariate(0.0, 0.4) + 0.9
        p = random()
        if p < 0.33:
            res[i].clustNum = 0
            res[i].x = 0.5 + r * cos(f)
            res[i].y = -0.25 + r * sin(f)
        elif p < 0.66:
            res[i].clustNum = 1
            res[i].x = -0.5 + r * cos(f)
            res[i].y = 0.25 - r * sin(f)
        else:
            res[i].clustNum = 2
            res[i].x = 0.0 + r * cos(f)
            res[i].y = 1 - r * sin(f)
    return res

class point2D:
    def __init__(self, x: float, y: float, clustNum: int) -> None:
        self.x = x
        self.y = y
        self.clustNum = clustNum

class cluster2D(point2D):
    def __init__(self, x: float, y: float, clustNum: int) -> None:
        super().__init__(x, y, clustNum)
        self.pointNum: int = 0
    
    def changePosition(self, points):
        self.pointNum = 0
        self.x = 0
        self.y = 0

        for p in points:
            if p.clustNum == self.clustNum:
                self.pointNum += 1
                self.x += p.x
                self.y += p.y
        self.x /= self.pointNum
        self.y /= self.pointNum


def EuclideanDistance(p1: point2D, p2: point2D) -> float:
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y))

def main() -> None:
    iterNum: int = 0
    colors = plt.cm.tab10(np.linspace(0, 1, CLUST_NUM))
    clusters = np.array([cluster2D(2*random()-1, 2*random()-1, i) for i in range(CLUST_NUM)])
    points = dataMoon()

    for p in points:
        plt.scatter(p.x, p.y, c='black', s=20)
    plt.show()

    while True:
        iterNum += 1
        clusterCopies = np.array([point2D(clusters[i].x, clusters[i].y, i) for i in range(CLUST_NUM)])

        for p in points:
            minDist: float = EuclideanDistance(clusters[0], p)
            clustNum: int = 0
            for i in range(1, CLUST_NUM):
                tmp: float = EuclideanDistance(clusters[i], p)
                if tmp < minDist:
                    minDist = tmp
                    clustNum = i
            p.clustNum = clustNum
            plt.scatter(p.x, p.y, c=colors[clustNum], s=20)

        for c in clusters: 
            c.changePosition(points)

        for c in clusters:
                plt.scatter(clusterCopies[c.clustNum].x, clusterCopies[c.clustNum].y, c='green', marker="*")
                plt.scatter(c.x, c.y, c='red', marker="*")
        plt.show()

        tmp: float = reduce(lambda x, y: x + y, [EuclideanDistance(clusters[i], clusterCopies[i]) for i in range(CLUST_NUM)], 0)
        if tmp < EPSILON: break
    print(f'Iteration numbers: {iterNum}')
    
main()