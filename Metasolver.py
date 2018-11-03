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
        self.pop_size = pop_size
        self.min_popsize = int(self.n*min_frac)

        self.reset()
    

    def reset(self):
        current_pop_ = [self.rg.choice(self.n,size=self.rg.randint(self.min_popsize,self.n+1), replace=False) for _ in range(self.pop_size)]
        self.current_pop = np.vstack([np.isin(list(range(self.n)),x) for x in current_pop_])
        self.current_solvers = [TSPSolver(graph=self.graph,vertices=x) for x in current_pop_]
        self.evalpop()
    
    @property
    def gen_size(self):
        return len(self.current_solvers)
    
    def evalpop(self):
        self.costs = np.zeros(self.gen_size)
        self.trajs = []
        for i in range(self.gen_size):
            soln = self.current_solvers[i].get_best_soln()
            self.trajs.append(soln[1])
            self.costs[i] = soln[0]
            self.costs[i] += np.sum(self.penalty[~self.current_pop[i]])
    
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
                new_ind[point] = ~new_ind[point]
        
        return new_ind
        


class MetaSolver(MetaPopulation):
    
    def __init__(self, graph, penalty, pop_size=50, mutation_rate=0.02, crossover_rate=0.1, seed=None, min_frac=0.6,
                fitness=None, selection_probab=None, max_pop=20,subiters=100, cut_frac=1.0):
        
        self.fitness = fitness
        self.selection_probab = selection_probab

        if self.fitness is None:
            self.fitness = lambda cost: np.exp(self.n*2/cost)
        if self.selection_probab is None:
            self.selection_probab = lambda fitness: fitness/np.sum(fitness)

        super(MetaSolver,self).__init__(graph, penalty, pop_size=pop_size, mutation_rate=mutation_rate, crossover_rate=crossover_rate, seed=seed, min_frac=min_frac)
        self.max_pop = self.pop_size
        self.subiters = subiters
        self.cut_frac=cut_frac
        self.evalpop()
        self.bestperf = []
    
    def one_step_train(self,subiters=None, debug=False):
        if subiters is None:
            subiters = self.subiters
        for solver in self.current_solvers:
            solver.train(iters=subiters, debug=debug)
    
    def evolve(self):
        #self.one_step_train(self.subiters)
        #pdb.set_trace()
        fitness = self.fitness(self.costs)
        bestsoln = np.argmax(fitness)
        select = self.selection_probab(fitness)
        select = select/select.sum()

        #Select population
        select_index = self.rg.choice(np.arange(self.gen_size),size=int(self.gen_size*self.cut_frac), replace=True, p=select)
        if bestsoln in select_index:
            self.current_pop = self.current_pop[select_index]
        else:
            self.current_pop = self.current_pop[np.append(select_index,bestsoln)]

        new_pop = self.current_pop.copy()
        for i in range(self.gen_size):
            ind = self.current_pop[i]
            ind2_index = self.rg.randint(self.gen_size)
            ind2 = self.current_pop[ind2_index]

            if self.rg.rand() < self.crossover_rate:
                new_ind1, new_ind2 = self.crossover(ind,ind2)
                if self.max_pop is None or self.gen_size<self.max_pop:
                    np.append(new_pop,[new_ind1], axis=0)
                    self.current_solvers.append(self.get_solver(new_ind1))
                    np.append(new_pop,[new_ind2], axis=0)
                    self.current_solvers.append(self.get_solver(new_ind2))
                else:
                    new_pop[i][:] = new_ind1
                    self.current_solvers[i] = self.get_solver(new_ind1)
                    new_pop[ind2_index][:] = new_ind2
                    self.current_solvers[ind2_index] = self.get_solver(new_ind2)

            
            new_pop[i,:] = self.mutate(ind)

        self.current_pop = new_pop
        self.evalpop()
    def get_best_soln(self):
        bestsoln = np.argmin(self.costs)
        return self.costs[bestsoln], self.trajs[bestsoln]
    
    def train(self,iters=500,plot=False,debug_2=False):
        for i in range(iters):
            self.one_step_train(self.subiters,debug_2)
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


        
