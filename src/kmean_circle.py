import matplotlib.pyplot as plt
from random import normalvariate, random
from math import sqrt, sin, cos

POINT_NUM = 500
CLUST_NUM = 3
EPSILON = 0.1
K_FACTOR = 3

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
    def __init__(self, cl, x, y, radius):
        super().__init__(cl, x, y)
        self.num = 0
        self.radius = radius

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

    def evalRadius(self, P):
        self.num = 0
        sigma = 0

        for p in P:
            if p.clustNum == self.clustNum:
                self.num += 1
                sigma += (self.x - p.x) * (self.x - p.x) + (self.y - p.y) * (self.y - p.y)

        if self.num > 0:
            self.radius = sqrt(K_FACTOR * sigma / self.num)

Cl = [cluster2D(i, 2.0*random()-1.0, 2.0*random()-1.0, 2.0*random() - 1.0) for i in range(CLUST_NUM)]

PP = [point2D(CLUST_NUM, 0.0, 0.0) for _ in range(POINT_NUM)]

dataIsland(PP, POINT_NUM)
#dataMoons(PP, POINT_N)

for p in PP: plt.scatter(p.x, p.y, c="black", s=20) # Source data  
plt.show()

colors = ["gray", "green", "purple"]
iterNum = 0
alpha = 0

for cl in Cl:
    cl.evalRadius(PP)

while (1):
    print(sum(1 for p in PP if p.clustNum == CLUST_NUM)) 
    iterNum += 1
    CC = [cluster2D(i, Cl[i].x, Cl[i].y, Cl[i].radius) for i in range(CLUST_NUM)]

    for cl in Cl:
        cl.evalCenter(PP)

    plt.gca().set_aspect('equal')
    
    for i in range(len(PP)):
        p = PP[i]

        minDist = float('inf')
        cln = None
        
        for k in range(CLUST_NUM):
            dist = Cl[k].EuclideanDist(p)
            if dist < minDist and dist < Cl[k].radius:
                minDist = dist
                cln = k
        
        if cln is not None:
            p.clustNum = cln
        else:
            p.clustNum = CLUST_NUM

        plt.scatter(p.x, p.y, color=colors[p.clustNum] if p.clustNum != CLUST_NUM else "black", s=20)


    for cl in Cl:
            circle = plt.Circle((cl.x, cl.y), cl.radius, color="red", fill=False)
            plt.gca().add_patch(circle)
            circle = plt.Circle((CC[cl.clustNum].x, CC[cl.clustNum].y), CC[cl.clustNum].radius, color="cyan", fill=False)
            plt.gca().add_patch(circle)
            plt.scatter(CC[cl.clustNum].x, CC[cl.clustNum].y, c="cyan", marker="*") # old
            plt.scatter(cl.x, cl.y, color="red", marker="*") # new
    plt.show()

    err = sum([Cl[i].EuclideanDist(CC[i]) for i in range(CLUST_NUM)])
    if (err) < EPSILON: break

print(f"Iteration number: {iterNum}")