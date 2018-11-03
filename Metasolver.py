from generate_graph import *
from graph import *
from GAtspSolver import TSPSolver
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pdb


class MetaPopulation(object):
    
    def __init__(self, graph,penalty,pop_size=50,
                mutation_rate=0.02,
                crossover_rate=0.1,seed=None,min_frac=0.6):
        self.graph = graph
        self.n = self.graph.n
        self.penalty = penalty
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.rg = np.random.RandomState(seed)
        self.min_popsize = int(self.n*min_frac)

        self.reset()
    

    def reset(self):
        current_pop_ = [self.rg.choice(self.n,size=self.rg.randint(self.min_popsize,self.n+1), replace=False)]
        self.current_pop = [np.isin(list(range(self.n)),x) for x in current_pop_]
        self.current_solvers = [TSPSolver(graph=self.graph,vertices=x) for x in current_pop_]
        self.evalpop()
    
    @property
    def gen_size(self):
        return len(self.current_solvers)
    
    def evalpop(self):
        self.costs = np.zeros(self.gen_size)
        for i in range(self.gen_size):
            self.costs[i] = self.current_solvers[i].get_best_soln()
    
    def get_solver(self,ind):
        v = np.arange(self.n)[ind]
        return TSPSolver(graph=self.graph, vertices=v)

    
    def crossover(self,ind1,ind2,crossover_point_rate = None):
        if crossover_point_rate is None:
            crossover_point_rate = self.crossover_rate
        cross_points = self.rg.binomial(1,crossover_point_rate,self.n).astype(np.bool)
        new_ind1 = ind1[:]
        new_ind1[cross_points] = ind2[cross_points]
        new_ind2 = ind2[:]
        new_ind2[cross_points] = ind1[cross_points]

        return new_ind1, new_ind2

    def mutate(self,ind,mutation_rate=None):
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        new_ind = ind[:]
        for point in np.arange(self.n):
            if self.rg.rand() < mutation_rate:
                swap_point = self.rg.randint(self.n)
                a,b = new_ind[point], new_ind[swap_point]
                new_ind[point], new_ind[swap_point] = b,a
        
        return new_ind
        




