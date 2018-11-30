from Polysolver import *
import numpy as np
from MetaRegressorSolver import MetaRegressor
from selection_functions import *
import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))

plt.ion()


xs = np.linspace(-2,2,50)
ys = xs**6 + (-5)*(xs**4)+ 4*(xs**2)+3*xs+4 + (0.01*np.random.randn(50))
#ys = 4*(xs**2)+3*xs+4 + (0.01*np.random.randn(50))
perm = np.random.permutation(50)

xs, ys = xs[perm], ys[perm]
xtrain, ytrain = xs[:40], ys[:40]
xtest, ytest = xs[40:], ys[40:]

def loss(y1,y, x=None):
    delta = 0.2
    l = np.abs(y1-y)
    idx1 = np.where(l>delta)
    idx2 = np.where(l<=delta)
    l[idx1] = delta*(l[idx1]-0.5*delta)
    l[idx2] = 0.5*(l[idx2]**2)
    idx3 = np.where(x>0)
    l[idx3] *= 100
    return np.mean(l)

solver_params={"mutation_rate":0.2, "crossover_rate":0.7, "pop_size":500, "min_pop":100,
        "selection_fun" : percentile,"cut_frac":1.0,"percentile":50}


mr = MetaRegressor(loss_fun=loss, ga=True, pop_size=100, max_deg=10, min_pop=20, solver_params=solver_params)
mr.init_solver(crossover_rate=0.5, selection_fun=percentile, percentile=60)
mr.train(xtrain,ytrain, xtest, ytest, iters=10, subiters=20, plot_curve=True)