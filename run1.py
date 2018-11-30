'''
Meta Solver training
'''

from GAtspSolver import TSPSolver
from MetaTSPsolver import MetaSolver
from graph import pointGraph, adjGraph
from generate_graph import *
from selection_functions import *
import matplotlib.pyplot as plt
from copy import deepcopy

plt.figure(figsize=(20,20))

#Get the graph with n=30 vertices
gr = pointGraph(n=30)
#ga = TSPSolver(graph=gr,cut_frac=0.6,initial_popsize=200, selection_fun=minmax, mutation_rate=0.02, crossover_rate=0.7, percentile=40)
#ga.train(iters=1000, plot=True, plotresult=True)

#Actual penalty
penalty = np.ones(30)*0.15 + np.random.rand(30)*0.2

#Hyperparameters for individual TSP solvers
tsp_params={"cut_frac":0.6, "selection_fun":minmax, "mutation_rate":0.02, "crossover_rate":0.7, "percentile":40}

#initialize Metasolver
ma = MetaSolver(graph=gr, penalty=penalty,subiters=30,sub_pop_size=200,pop_size=30,
    selection_fun=percentile,
    fitness = lambda x:np.exp(-x), crossover_rate=0.5, min_frac=0.2, mutation_rate=0.2, tsp_params=tsp_params)

#Train metasolver
ma.train(iters=30, plot=True)

train1 = deepcopy(ma.bestperf)

#NOW WE DO ADAPTIVE TRAINING

#Initialize metasolver
ma = MetaSolver(graph=gr, penalty=penalty,subiters=30,sub_pop_size=200,pop_size=30,
    selection_fun=percentile,
    fitness = lambda x:np.exp(-x), crossover_rate=0.5, min_frac=0.2, mutation_rate=0.2, tsp_params=tsp_params)


#Initial penalty is maximum for all nodes
penalty_curr = np.ones(30)*np.max(penalty)
diff = penalty_curr - penalty

superiters = 10
diff = diff/(superiters-1)

for i in range(superiters):
    ma.penalty = penalty_curr
    #Train
    ma.train(iters=2, plot=True, penalty=penalty)
    #Update penalty
    penalty_curr = penalty_curr - diff

#Unadapted training
ma.penalty = penalty
ma.train(iters=10, plot=True)

train2 = deepcopy(ma.bestperf)

plt.clf()
plt.xlabel('Genrations')
plt.ylabel('Cost')
xs = np.arange(30)+1
plt.plot(xs, train1, label="Non-adaptive")
plt.plot(xs, train2, label="Adaptive")
plt.legend()