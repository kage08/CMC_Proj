from solver import *
import numpy as np
from GASolver import MetaRegressor
from selection_functions import *
import matplotlib.pyplot as plt

plt.ion()

xs = np.linspace(0,50,50)
ys = 2*(xs**2)+3*xs+4 + (0.01*np.random.randn(50))
perm = np.random.permutation(50)

xs, ys = xs[perm], ys[perm]
xtrain, ytrain = xs[:40], ys[:40]
xtest, ytest = xs[40:], ys[40:]

def loss(y1,y):
    return np.mean(np.square(y1-y))


mr = MetaRegressor(loss_fun=loss)
mr.init_solver(crossover_rate=0.3, selection_fun=percentile, percentile=60)
mr.train(xtrain,ytrain, xtest, ytest, iters=10, subiters=50)