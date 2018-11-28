from GAtspSolver import TSPSolver
from MetaTSPsolver import MetaSolver
from graph import pointGraph, adjGraph
from generate_graph import *
from selection_functions import *
import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))

gr = pointGraph(n=40)
#ga = TSPSolver(graph=gr,cut_frac=1.0,initial_popsize=500, selection_fun=minmax)
#ga.train(iters=500, plot=True, plotresult=True)

penalty = np.ones(40)*10
tsp_params={}
ma = MetaSolver(graph=gr, penalty=penalty,subiters=50,sub_pop_size=2000,pop_size=10,
    selection_fun=percentile,
    fitness = lambda x:np.exp(-x), crossover_rate=0.5, min_frac=0.7, mutation_rate=0.2, tsp_params=tsp_params)