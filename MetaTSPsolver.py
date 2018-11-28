from generate_graph import *
from graph import *
from GAtspSolver import TSPSolver
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pdb
from selection_functions import minmax


class MetaPopulation(object):
    
    def __init__(self, graph,penalty,pop_size=50,
                mutation_rate=0.02,
                crossover_rate=0.1,seed=None,min_frac=0.6, sub_pop_size=500, tsp_params = {}):
        self.graph = graph
        self.n = self.graph.n
        self.penalty = penalty
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.rg = np.random.RandomState(seed)
        self.pop_size = pop_size
        self.min_popsize = int(self.n*min_frac)
        self.sub_pop_size = sub_pop_size
        self.tsp_params = tsp_params
        self.reset()
    

    def reset(self):
        self.current_pop_ = [self.rg.choice(self.n,size=self.rg.randint(self.min_popsize,self.n+1), replace=False) for _ in range(self.pop_size)]
        self.current_pop = np.vstack([np.isin(list(range(self.n)),x) for x in self.current_pop_])
        self.current_solvers = [TSPSolver(graph=self.graph,vertices=x,initial_popsize=self.sub_pop_size, **self.tsp_params) for x in self.current_pop_]
        self.evalpop()
    
    @property
    def gen_size(self):
        return len(self.current_pop)
    
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
                fitness=None, selection_fun=None, max_pop=20,subiters=100, cut_frac=1.0,sub_pop_size=500, percentile=50, tsp_params={}, min_pop_frac = 0.3):
        
        self.fitness = fitness
        self.selection_fun = selection_fun
        self.min_pop_frac = min_pop_frac

        if self.fitness is None:
            self.fitness = lambda cost: np.exp(self.n*2/cost)
        if self.selection_fun is None:
            self.selection_fun = minmax
        
        self.percentile = percentile

        super(MetaSolver,self).__init__(graph, penalty, pop_size=pop_size, mutation_rate=mutation_rate, crossover_rate=crossover_rate, seed=seed, min_frac=min_frac, sub_pop_size=sub_pop_size, tsp_params=tsp_params)
        self.max_pop = self.pop_size
        self.subiters = subiters
        self.cut_frac=cut_frac
        self.evalpop()
        self.bestperf = []

        self.debug = False
    
    def one_step_train(self,subiters=None, debug=False):
        if subiters is None:
            subiters = self.subiters
        for solver in self.current_solvers:
            solver.train(iters=subiters, debug=debug)
    
    def evolve(self):
        #self.one_step_train(self.subiters)
        if self.debug:
            pdb.set_trace()
        fitness = self.fitness(self.costs)
        bestsoln = np.argmax(fitness)
        

        #Select population
        #select_index = self.rg.choice(np.arange(self.gen_size),size=int(self.gen_size*self.cut_frac), replace=True, p=select)
        select_index = self.selection_fun(fitness=fitness,gen_size=self.gen_size, cut_frac=self.cut_frac, percentile=self.percentile)
        if not (bestsoln in select_index):
            select_index = np.append(select_index,bestsoln)
        
        self.current_pop = self.current_pop[select_index]
        self.current_solvers = [self.current_solvers[i] for i in select_index]

        self.evalpop()
        fitness = self.fitness(self.costs)
        bestsoln = np.argmax(fitness)

        new_pop = self.current_pop.copy()
        for i in range(self.gen_size):
            ind = self.current_pop[i]
            ind2_index = self.rg.randint(self.gen_size)
            ind2 = self.current_pop[ind2_index]

            if self.rg.rand() < self.crossover_rate:
                new_ind1, new_ind2 = self.crossover(ind,ind2)
                if self.max_pop is None or self.gen_size<self.max_pop:
                    new_pop = np.append(new_pop,[new_ind1], axis=0)
                    self.current_solvers.append(self.get_solver(new_ind1))
                    new_pop = np.append(new_pop,[new_ind2], axis=0)
                    self.current_solvers.append(self.get_solver(new_ind2))
                elif i != bestsoln:
                    new_pop[i,:] = new_ind1
                    self.current_solvers[i] = self.get_solver(new_ind1)
                    new_pop[ind2_index,:] = new_ind2
                    self.current_solvers[ind2_index] = self.get_solver(new_ind2)

            if i != bestsoln:
                new_pop[i,:] = self.mutate(ind)
                self.current_solvers[i] = self.get_solver(new_pop[i])
        

        self.current_pop = new_pop



        # Maintain above min population size
        while self.gen_size <= self.max_pop*self.min_pop_frac:
            ind2_index = self.rg.randint(self.gen_size)
            ind = self.current_pop[bestsoln]
            ind2 = self.current_pop[ind2_index]
            new_ind1, new_ind2 = self.crossover(ind,ind2)
            self.current_pop = np.append(self.current_pop,[new_ind1],axis=0)
            self.current_solvers.append(self.get_solver(new_ind1))
            self.current_pop = np.append(self.current_pop,[new_ind2],axis=0)
            self.current_solvers.append(self.get_solver(new_ind2))

            new_ind1 = self.mutate(self.current_pop[bestsoln])
            self.current_pop = np.append(self.current_pop,[new_ind1],axis=0)
            self.current_solvers.append(self.get_solver(new_ind1))

        self.evalpop()
        
    def get_best_soln(self, plot=False):
        bestsoln = np.argmin(self.costs)
        c,t = self.costs[bestsoln], self.trajs[bestsoln]
        if plot:
            self.graph.plot(t,c)
        return c,t
    
    def train(self,iters=500,plot=False,debug_2=False):
        for i in range(iters):
            self.one_step_train(self.subiters,debug_2)
            self.evolve()
            best = self.get_best_soln()
            self.bestperf.append(best[0])
            print("Gen:",str(i+1),"Best Cost:", best[0])
            if plot:
                self.graph.plot(best[1],best[0])
        
        #plt.pause(10)
        plt.clf()
        plt.plot(np.arange(len(self.bestperf)),self.bestperf)


        
