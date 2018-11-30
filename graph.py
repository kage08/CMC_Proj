from generate_graph import *
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class Graphs(object):
    
    def __init__(self,n,m=None):
        self.n = n
        self.m = m
        self.adj=None
    
    def get_dist(self,u,v):
        raise NotImplementedError("Not implemented distance metric")
    
    def get_adj(self):
        raise NotImplementedError("Not implemented adjacency matrix")


class pointGraph(Graphs):

    def __init__(self, n, m=None, parray=None):
        self.parray = parray

        if self.parray is None:
            self.parray = np.random.rand(n,2)
        super(pointGraph,self).__init__(n, m=m)
        self.get_adj()
        plt.ion()
        
    

    def get_dist1(self, u, v):
        return np.sqrt(np.square(self.parray[u,0]-self.parray[v,0])+np.square(self.parray[u,1]-self.parray[v,1]))
    
    def get_dist(self,u,v):
        return self.adj[u,v]
    
    def get_adj(self):
        self.adj = np.zeros((self.n,self.n))
        for i in range(self.n-1):
            for j in range(i+1,self.n):
                self.adj[i,j] = self.adj[j,i] = self.get_dist1(i,j)
        return self.adj
    
    def plot(self,path,path_dist, penalty=None):
        plt.cla()
        plt.scatter(self.parray[:,0],self.parray[:,1], s=100, c=penalty)
        xs = [self.parray[i,0] for i in path]
        ys = [self.parray[i,1] for i in path]
        plt.plot(xs,ys)
        plt.text(-0.05, -0.05, "Total distance=%.2f" % path_dist, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.01)


class adjGraph(Graphs):
    def __init__(self, adjmatrix,n=None, m=None, parray=None):
        self.adj = np.array(adjmatrix)
        n = self.adj.shape(0)
        super().__init__(n, m=m)
    
    def get_dist(self, u, v):
        return self.adj[u,v]
    
    def get_adj(self):
        return self.adj