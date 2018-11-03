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
                crossover_rate=0.1,seed=None):
        self.graph = graph
        self.n = self.graph.n
        self.penalty = penalty
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.rg = np.random.RandomState(seed)
    

    def reset(self):
        self.current_pop = []



