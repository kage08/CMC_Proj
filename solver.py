import numpy as np
import tensorflow as tf

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
    
    
 