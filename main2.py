import numpy as np
from Polysolver import PolyGASolver as PGS
import matplotlib.pyplot as plt
from selection_functions import *

plt.ion()

xs = np.linspace(-2,2,50)
ys = xs**6 + (-5)*(xs**4)+ 4*(xs**2)+3*xs+4 + (0.01*np.random.randn(50))
perm = np.random.permutation(50)

xs, ys = xs[perm], ys[perm]
xtrain, ytrain = xs[:40], ys[:40]
xtest, ytest = xs[40:], ys[40:]

pgs = PGS(0,0.5,6, selection_fun=percentile, percentile=40)
pgs.evalpop(xtest, ytest)
pgs.initialize()