from GAtspSolver import TSPSolver
from graph import pointGraph, adjGraph
from generate_graph import *

gr = pointGraph(n=20)
ga = TSPSolver(graph=gr,cut_frac=1.0,initial_popsize=500, selection_probab=lambda x: x>np.median(x))
ga.train(iters=500)