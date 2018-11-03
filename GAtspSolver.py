from generate_graph import *
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pdb



class TSPPopulation(object):

    def __init__(self, 
                graph,seed=None,vertices=None,
                initial_popsize = 500,
                mutation_rate=0.02,
                crossover_point_rate = 0.5):
        self.graph = graph
        self.vertices = vertices
        self.n = len(vertices)
        self.rg = np.random.RandomState(seed)

        self.mutation_rate = mutation_rate
        
        self.crossover_point_rate = crossover_point_rate

        self.pop_size = initial_popsize

        self.reset()

        
    @property
    def gen_size(self):
        return self.current_pop.shape[0]
    
    def reset(self):
        self.current_pop = np.vstack([self.rg.permutation(self.vertices) for _ in range(self.pop_size)])
        self.costs = np.empty((len(self.current_pop)))
        self.evalpop()
    
    
    def evalpop(self):
        self.costs = np.zeros(self.gen_size)
        for i, p in enumerate(self.current_pop):
            for j in range(self.n-1):
                self.costs[i] += self.graph.get_dist(p[j],p[j+1])
            #self.costs[i] += self.graph.get_dist(p[0],p[-1])
        
    def mutate(self,ind,mutate_rate=None, copy_ind=False):
        if mutate_rate is None:
            mutate_rate = self.mutation_rate
        if copy_ind:
            new_ind = deepcopy(ind)
        else:
            new_ind = ind
        
        for point in range(self.n):
            if self.rg.rand() < mutate_rate:
                swap_point = self.rg.randint(self.n)
                a,b = ind[point], ind[swap_point]
                new_ind[point], new_ind[swap_point] = b,a
        
        return new_ind
            
    def crossover(self, ind1, ind2, crossover_point_rate = None):
        if crossover_point_rate is None:
            crossover_point_rate = self.crossover_point_rate
        
        cross_points = self.rg.binomial(1,crossover_point_rate,self.n).astype(np.bool)
        keep_points = ind1[~cross_points]
        swap_points = ind2[np.isin(ind2,keep_points,invert=True)]
        new_ind = np.concatenate((keep_points,swap_points))

        return new_ind


class TSPSolver(TSPPopulation):
    def __init__(self,graph,crossover_rate = 0.1,fitness=None,
                selection_probab = None,cut_frac=0.8, *args, **kwargs):

        self.crossover_rate = crossover_rate
        self.fitness = fitness
        self.cut_frac = cut_frac
        if self.fitness is None:
            self.fitness = lambda cost: np.exp(self.n*2/cost)
        
        self.selection_probab = selection_probab
        if self.selection_probab is None:
            self.selection_probab = lambda fitness: fitness/np.sum(fitness)
        super(TSPSolver,self).__init__(graph=graph,*args, **kwargs)
        self.max_pop=self.pop_size
        self.evalpop()
        self.bestperf = []

    
    def evolve(self):
        fitness = self.fitness(self.costs)
        bestsoln = np.argmax(fitness)
        select = self.selection_probab(fitness)
        select = select/select.sum()

        #pdb.set_trace()
        #Select population
        select_index = self.rg.choice(np.arange(self.gen_size),size=int(self.gen_size*self.cut_frac), replace=True, p=select)
        if bestsoln in select_index:
            self.current_pop = self.current_pop[select_index]
        else:
            self.current_pop = self.current_pop[np.append(select_index,bestsoln)]

        new_pop = self.current_pop.copy()
        for i in range(self.gen_size):
            ind = self.current_pop[i]
            ind2 = self.rg.randint(self.gen_size)
            ind2 = self.current_pop[ind2]

            if self.rg.rand() < self.crossover_rate:
                new_ind = self.crossover(ind,ind2)
                if self.max_pop is None or self.gen_size<self.max_pop:
                    np.append(new_pop,[new_ind], axis=0)
                else:
                    new_pop[i,:] = new_ind
            
            new_pop[i,:] = self.mutate(ind)
        
        self.current_pop = new_pop
        self.evalpop()

    
    def get_best_soln(self):
        fitness = self.fitness(self.costs)
        bestsoln = np.argmin(self.costs)
        return self.costs[bestsoln], self.current_pop[bestsoln]
    
    def train(self, iters=500, plot=False):
        for i in range(iters):
            self.evolve()
            best = self.get_best_soln()
            self.bestperf.append(best[0])
            print("Gen:",str(i+1),"Best Cost:", best[0])
            if plot:
                self.graph.plot(best[1],best[0])
        
        plt.pause(10)
        plt.ioff()
        plt.clf()
        plt.plot(np.arange(len(self.bestperf)),self.bestperf)
        plt.show()
        


