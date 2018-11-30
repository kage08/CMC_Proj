'''
Simulate different objective functions
'''

import numpy as np
import matplotlib.pyplot as plt

plt.ion()
plt.figure(figsize=(20,20))
plt.clf()
plt.xlabel('Prediction')
plt.ylabel('Loss')

xs = np.linspace(-10,10,1000)


mse = 0.5*(xs**2)
mae = np.abs(xs)
#plt.plot(xs,mse,label='MSE')


delta = 0.5
idx1 = np.abs(xs)< delta
huber_1 = xs.copy()
huber_1[idx1] = mse[idx1]
huber_1[~idx1] = delta*mae[~idx1] - 0.5*(delta**2)
plt.plot(xs,huber_1,label='Huber delta=0.5')

delta = 1
idx1 = np.abs(xs)< delta
huber_2 = xs.copy()
huber_2[idx1] = mse[idx1]
huber_2[~idx1] = delta*mae[~idx1] - 0.5*(delta**2)
plt.plot(xs,huber_2,label='Huber delta=1')

gamma = 0.25
quantile_1 = np.max([gamma*xs, (gamma-1)*xs], axis=0)
plt.plot(xs,quantile_1, label='quantile loss gamma=0.25')
gamma = 0.75
quantile_2 = np.max([gamma*xs, (gamma-1)*xs], axis=0)
plt.plot(xs,quantile_2, label='quantile loss gamma=0.75')



plt.legend()
