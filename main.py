from GAtspSolver import TSPSolver
from graph import pointGraph, adjGraph
from generate_graph import *

gr = pointGraph(n=20)
ga = TSPSolver(graph=gr,vertices=[4,5,6,7,9,10],cut_frac=1.0,initial_popsize=500, selection_probab=lambda x: x>np.median(x))
ga.train(iters=500)