import matplotlib.pyplot as plt
from random import normalvariate, random
from math import sqrt, sin, cos
import numpy as np
from matplotlib.patches import Ellipse

POINT_NUM = 500
CLUST_NUM = 3
EPSILON = 0.1
K_FACTOR = 3

def dataEllipses(A, N):
    for i in range(N):
        p = random()
        
        a, b = 1.5, 0.4 
        if p < 0.33:
            A[i].clustNum = 0
            angle = 0
            xx = normalvariate(0, a*0.8)
            yy = normalvariate(0, b*0.8)
            A[i].x = -2.0 + xx * cos(angle) - yy * sin(angle)
            A[i].y = 0.0 + xx * sin(angle) + yy * cos(angle)
        elif p < 0.66:
            A[i].clustNum = 1
            angle = 90 * 3.14/180
            xx = normalvariate(0, a*0.8)
            yy = normalvariate(0, b*0.8)
            A[i].x = 2.0 + xx * cos(angle) - yy * sin(angle)
            A[i].y = 0.0 + xx * sin(angle) + yy * cos(angle)
        else:
            A[i].clustNum = 2
            angle = 45 * 3.14/180
            xx = normalvariate(0, a*0.8)
            yy = normalvariate(0, b*0.8)
            A[i].x = 0.0 + xx * cos(angle) - yy * sin(angle)
            A[i].y = 2.0 + xx * sin(angle) + yy * cos(angle)

def dataIsland(A, N):
    for i in range(N):
        p = random()

        if p < 0.33:
            A[i].clustNum = 0
            A[i].x = normalvariate(-1.0, 0.5)
            A[i].y = normalvariate(0.0, 1.0)
        elif p < 0.66:
            A[i].clustNum = 1
            A[i].x = normalvariate(1.0, 0.5)
            A[i].y = normalvariate(0.0, 0.75)
        else: 
            A[i].clustNum = 2
            A[i].x = normalvariate(1.0, 0.4)
            A[i].y = normalvariate(0.0, 0.75)

def dataMoons(A, N):
    for i in range(N):
        f = 3.14 * random()
        r = 0.2 * normalvariate(0.0, 0.4) + 0.9
        p = random()
        
        if p < 0.33:
            A[i].clustNum = 0
            A[i].x = 0.5 + r * cos(f)
            A[i].y = -0.25 + r * sin(f)
        elif p < 0.66:
            A[i].clustNum = 1
            A[i].x = -0.5 + r * cos(f)
            A[i].y = 0.25 - r * sin(f)
        else:
            A[i].clustNum = 2
            A[i].x = -1.0 + r * cos(f)
            A[i].y = -0.25 - r * sin(f)

class point2D:
    def __init__(self, cl, x, y):
        self.clustNum = cl
        self.x = x
        self.y = y

    def EuclideanDist(self, p):
        return sqrt((self.x - p.x)*(self.x - p.x) + (self.y - p.y)*(self.y - p.y))

class cluster2D(point2D):
    def __init__(self, cl, x, y):
        super().__init__(cl, x, y)
        self.num = 0

        self.a = 1.0
        self.b = 1.0
        self.angle = 0

        self.covMrx = None
        self.invCovMrx = None

    def evalCenter(self, P):
        self.num = 0
        self.x = 0.0
        self.y = 0.0
        for p in P:
            if p.clustNum == self.clustNum:
                self.num += 1
                self.x += p.x
                self.y += p.y
        if self.num > 0:
            self.x /= self.num
            self.y /= self.num

    def evalCovMrx(self, P):
        if self.num < 2: return
        
        points = []
        for p in P:
            if p.clustNum == self.clustNum:
                points.append([p.x, p.y])
        
        points = np.array(points)
        self.covMrx = np.cov(points.T)
        
        self.invCovMrx = np.linalg.inv(self.covMrx) 

        vals, vecs = np.linalg.eig(self.covMrx)
        self.a = sqrt(vals[0] * K_FACTOR)
        self.b = sqrt(vals[1] * K_FACTOR)
        self.angle = np.arctan2(vecs[1, 0], vecs[0, 0])


    def MahalanobisDist(self, p):
        diff = np.array([p.x - self.x, p.y - self.y])
        return sqrt(diff.dot(self.invCovMrx).dot(diff))
    
    def containPoint(self, p):
        return (p.x - self.x) * (p.x - self.x) / (self.a * self.a) + (p.y - self.y) * (p.y - self.y) / (self.b * self.b) <= 1


Cl = [cluster2D(i, 2.0*random()-1.0, 2.0*random()-1.0) for i in range(CLUST_NUM)]

PP = [point2D(CLUST_NUM, 0.0, 0.0) for _ in range(POINT_NUM)]

#dataIsland(PP, POINT_NUM)
#dataMoons(PP, POINT_NUM)
dataEllipses(PP, POINT_NUM)

for p in PP: plt.scatter(p.x, p.y, c="black", s=20) # Source data  
plt.show()

colors = ["gray", "green", "purple"]
iterNum = 0

while (1):
    iterNum += 1
    CC = [point2D(i, Cl[i].x, Cl[i].y) for i in range(CLUST_NUM)]

    for cl in Cl:
        cl.evalCenter(PP)
        cl.evalCovMrx(PP)
    
    for i in range(len(PP)):
        p = PP[i]

        minDist = Cl[p.clustNum].MahalanobisDist(p)
        cln = p.clustNum
        
        for k in range(CLUST_NUM):
            dist = Cl[k].MahalanobisDist(p)
            if dist < minDist and Cl[k].containPoint(p):
                minDist = dist
                cln = k

        p.clustNum = cln

        plt.scatter(p.x, p.y, color=colors[p.clustNum] if p.clustNum != CLUST_NUM else "black", s=20)

    for cl in Cl:
        ellipse = Ellipse(xy=(cl.x, cl.y), 
                             width=2*cl.a, 
                             height=2*cl.b,
                             angle=np.degrees(cl.angle),
                             color="red", fill=False, linewidth=2)
        plt.gca().add_patch(ellipse)
        plt.scatter(CC[cl.clustNum].x, CC[cl.clustNum].y, c="cyan", marker="*") # old
        plt.scatter(cl.x, cl.y, color="red", marker="*") # new
    plt.show()

    err = sum([Cl[i].MahalanobisDist(CC[i]) for i in range(CLUST_NUM)])
    if (err) < EPSILON: break

print(f"Iteration number: {iterNum}")