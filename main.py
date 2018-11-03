from GAtspSolver import TSPSolver
from Metasolver import MetaSolver
from graph import pointGraph, adjGraph
from generate_graph import *

gr = pointGraph(n=7)
#ga = TSPSolver(graph=gr,vertices=[4,5,6,7,9,10],cut_frac=1.0,initial_popsize=500, selection_probab=lambda x: x>np.median(x))
#ga.train(iters=500)
penalty = np.ones(7)*1.5
ma = MetaSolver(graph=gr, penalty=penalty,subiters=5,selection_probab=lambda x: x>np.median(x), fitness = lambda x:np.exp(-x))