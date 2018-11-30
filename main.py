from GAtspSolver import TSPSolver
from MetaTSPsolver import MetaSolver
from graph import pointGraph, adjGraph
from generate_graph import *
from selection_functions import *
import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))

gr = pointGraph(n=40)
#ga = TSPSolver(graph=gr,cut_frac=0.6,initial_popsize=200, selection_fun=minmax, mutation_rate=0.02, crossover_rate=0.7, percentile=40)
#ga.train(iters=1000, plot=True, plotresult=True)

penalty = np.ones(40)*0.3
tsp_params={"cut_frac":0.6, "selection_fun":minmax, "mutation_rate":0.02, "crossover_rate":0.7, "percentile":40}
ma = MetaSolver(graph=gr, penalty=penalty,subiters=200,sub_pop_size=200,pop_size=30,
    selection_fun=percentile,
    fitness = lambda x:np.exp(-x), crossover_rate=0.5, min_frac=0.2, mutation_rate=0.2, tsp_params=tsp_params)

ma.train(iters=30, plot=True)