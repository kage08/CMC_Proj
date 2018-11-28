import numpy as np
import tensorflow as tf
from selection_functions import minmax
from selection_functions import percentile as perc
from copy import deepcopy
import pdb
import matplotlib.pyplot as plt

class PolynomialSolver(object):

    def __init__(self, l2, gamma, degree, l1=0):
        self.l1 = l1
        self.l2 = l2
        self.gamma = gamma
        self.degree = degree

        self.model()
        self.losses()

    def model(self):
        self.x = tf.placeholder("float")
        self.y = tf.placeholder("float")
        self.w = [tf.Variable(np.random.randn())]
        self.out = self.w[0]
        #temp = self.x
        for i in range(self.degree):
            self.w.append(tf.Variable(np.random.randn()))
            self.out = self.out + (self.w[-1]*tf.pow(self.x,i+1))
    
    def losses(self, lr=0.01):
        self.error = self.out-self.y
        self.mse_loss = tf.reduce_mean(tf.pow(self.error,2))
        self.coshloss = tf.reduce_mean(tf.math.log(tf.math.cosh(self.error)))
        self.quantileloss = tf.reduce_mean(tf.maximum(self.gamma*self.error, (self.gamma-1)*self.error))

        self.loss = self.mse_loss  + self.l2*self.quantileloss + self.l1*tf.reduce_sum(tf.square(self.w))

        self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def initialize(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def fit(self,xtrain,ytrain, epochs=100, debug=False):

        for ep in range(epochs):
            for (x,y) in zip(xtrain, ytrain):
                self.sess.run(self.optimizer, feed_dict={self.x:x, self.y:y})
            if debug and (ep+1)%10==0:
                print("Epoch:",ep+1,", Loss:",self.sess.run(self.loss, feed_dict={self.x:xtrain, self.y:ytrain}))
    

    def predict(self, xtest):
        return self.sess.run(self.out,feed_dict={self.x:xtest})
    
    def close(self):
        self.sess.close()
    
    
class PolyGASolver(object):
    def __init__(self, l2, gamma, degree, l1=0, mutation_rate=0.2, crossover_rate=0.4, pop_size=500, min_pop=100,
     crossover_point_rate=0.5, seed=None, mean=0, stdev=10, noise_stdev=2):
        self.l1 = l1
        self.l2 = l2
        self.gamma = gamma
        self.degree = degree

        self.pop_size = pop_size
        self.min_pop = min_pop
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.crossover_point_rate = crossover_point_rate
        self.rg = np.random.RandomState(seed)

        self.mean = mean
        self.stdev = stdev
        self.noise = noise_stdev

        self.reset()
    

    def get_new_individual(self):
        return [self.rg.normal(self.mean, self.stdev) for i in range(self.degree+1)]
    
    def reset(self):
        self.current_pop = np.array([self.get_new_individual() for i in range(self.pop_size)])
    
    #def get_error(self, indiv, ypred, ytest):
    #    mse_error = 
    
    def evalpop(self, xtrain, ytrain):
        self.ytrain = np.array(ytrain) 
        self.ypred = self.predict(xtrain)
        self.error = self.ypred - self.ytrain

        self.mse_loss = np.mean(self.error**2,axis=1)
        self.quantile_loss = np.mean(np.max([self.gamma*self.error, (self.gamma-1)*self.error], axis=0), axis=1)

        self.l2_norm = np.sum(self.current_pop**2, axis=1)

        self.costs = self.mse_loss  + self.l2*self.quantile_loss + self.l1*self.l2_norm
    
    @property
    def gen_size(self):
        return self.current_pop.shape[0]
    
    def mutate(self, ind_):
        ind = ind_.copy()
        for i in range(self.degree+1):
            ind[i] += self.noise*self.rg.randn()
        return ind
    
    def crossover(self, ind1_, ind2_):
        ind1 = ind1_.copy()
        ind2 = ind2_.copy()

        for i in range(self.degree+1):
            if self.rg.rand() < self.crossover_point_rate: ind1[i], ind2[i] = ind2[i], ind1[i]
        
        return ind1, ind2
    
    def initialize(self,fitness=None,
                selection_fun = perc,cut_frac=1.0,percentile=50):
    
        self.fitness = fitness
        self.cut_frac = cut_frac
        self.percentile = percentile

        if self.fitness is None:
            self.fitness = lambda cost: -cost

        self.selection_fun = selection_fun
        if self.selection_fun is None:
            self.selection_fun = minmax
        
        self.max_pop=self.pop_size
        self.bestperf = []
    
    def evolve(self, x, y):
        #pdb.set_trace()
        self.evalpop(x, y)
        fitness = self.fitness(self.costs)
        bestsoln = np.argmax(fitness)

        select_index = self.selection_fun(fitness=fitness,gen_size=self.gen_size, cut_frac=self.cut_frac, percentile=self.percentile)
        if not (bestsoln in select_index):
            select_index = np.append(select_index,bestsoln)
        
        self.current_pop = self.current_pop[select_index]

        self.evalpop(x, y)
        fitness = self.fitness(self.costs)
        bestsoln = np.argmax(fitness)

        new_pop = self.current_pop.copy()

        for i in range(self.gen_size):
            ind = self.current_pop[i]
            ind2_idx = self.rg.randint(self.gen_size)
            ind2 = self.current_pop[ind2_idx]

            if self.rg.rand() < self.crossover_rate:
                
                new_ind1, new_ind2 = self.crossover(ind,ind2)
                if self.max_pop is None or self.gen_size<self.max_pop:
                    new_pop = np.append(new_pop,[new_ind1, new_ind2], axis=0)
                elif i != bestsoln:
                    #self.solvers[i].close()
                    #self.solvers[ind2_idx].close()
                    new_pop[i] = new_ind1[:]
                    new_pop[ind2_idx] = new_ind2[:]
                
            if i != bestsoln and self.rg.rand() < self.mutation_rate:
                new_ind = self.mutate(ind)
                new_pop[i] = new_ind[:]
        
        self.current_pop = new_pop
        self.evalpop(x, y)
    
    def get_best_soln(self):
        bestsoln = np.argmin(self.costs)
        return self.costs[bestsoln], self.current_pop[bestsoln]
    
    def fit(self, xtrain, ytrain, epochs=500, plotresult=False, debug=False):
        for i in range(epochs):
            self.evolve(xtrain, ytrain)
            best = self.get_best_soln()
            self.bestperf.append(best[0])
            if debug:
                print("Gen:",str(i+1),"Best Cost:", best[0])
            
        if plotresult:
            #plt.pause(10)
            #plt.ioff()
            plt.clf()
            plt.plot(np.arange(len(self.bestperf)),self.bestperf)
        
    def predict(self, xtrain):
        self.xtrain = np.array(xtrain)
        xtemp = np.array([self.xtrain**i for i in range(self.degree+1)])
        self.ypred = self.current_pop.dot(xtemp)
        return self.ypred

