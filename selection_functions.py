'''
Different selection strategies
'''

import numpy as np

def minmax(**kwargs):
    fitness = kwargs['fitness']
    prob = (fitness-np.min(fitness))/(np.max(fitness)-np.min(fitness))
    prob = prob/np.sum(prob)
    select_index = np.random.choice(np.arange(kwargs['gen_size']),size=int(kwargs['gen_size']*kwargs['cut_frac']), replace=True, p=prob)
    return select_index

def softmax(**kwargs):
    fitness = kwargs['fitness']
    expo = np.exp(fitness)
    prob = expo/np.sum(expo)
    select_index = np.random.choice(np.arange(kwargs['gen_size']),size=int(kwargs['gen_size']*kwargs['cut_frac']), replace=True, p=prob)
    return select_index


def percentile(**kwargs):
    fitness = kwargs['fitness']
    p = kwargs['percentile']
    return np.where(fitness>np.percentile(fitness,p))[0]
    
def random(**kwargs):
    return np.random.choice(np.arange(kwargs['gen_size']),size=int(kwargs['gen_size']*kwargs['cut_frac']), replace=False)