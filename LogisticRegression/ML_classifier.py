# *******************************************************************************
# **
# **  Author: Michael Lomnitz (mrlomnitz@lbl.gov)                               
# **  Python module holding logistic regression classifier authored by M. Lomnitz
# **
# *******************************************************************************

# Import relevant modules
import numpy as np
import math as m

class ML_classifier(object):
    # -- - - Makes eigne vector from data label
    def make_eigen( self, label ):
        _Y = np.zeros(10)
        print(label)
        _Y[label] =1 
        return _Y
    
    # -- - - Distance (i.e. entropy loss)
    def distance( self, X_, Y_ ):
        temp = np.dot( self.W, X_.T ) + self.B - Y_
        return m.sqrt( np.dot( temp, temp.T ) )

    # -- - - Softmax to convert discreet labels into probabilities
    def soft_max( _pred):
        y = np.exp( _pred )
        return y/np.sum(y)

    # -- - - Caulclate gradient for weights and biasses
    def gradient(self, X_, Y_,):
        _C = self.B - Y_
        _A = np.dot(self.W,X_.T)
        # -- - - Bias
        grad_B = 2.*(_A+_C)
        # -- - - Weights
        grad_W = np.array( 2.*X_[0]*(_A+_C).T )
        for idx in range(1,len(X_)):
            grad_W = np.hstack( (grad_W,2.*X_[idx]*(_A+_C).T) )
        # -- - - 
        return grad_W,grad_B

    # -- - - Minimization( i.e. training ) 
    def minimize(self, data, label):
        # -- - - initialize classifier weights random, bias to zero
        self.W = np.random.standard_normal( (label.shape[1],data.shape[1]) )
        self.B = np.zeros( (1, label.shape[1]) )
        # -- - - Local variables for minimization
        tot_dist = 999
        gradient = 0
        N = len(data)
        step = 0.001/N
        # -- - - Minimization steps
        for idx in range(51):
            entropy_loss = 0 
            delta_w = np.zeros( self.W.shape )
            delta_b = np.zeros( self.B.shape )
            # -- - - Loop over data
            for _X, _label in zip(data, label):
                #_Y = self.make_eigen( _label )
                _Y = _label
                gW,gB = self.gradient(_X, _Y)
                delta_w += gW
                delta_b += gB
                entropy_loss+=self.distance(_X, _Y)
            # update weights and bias
            self.W = self.W - step*delta_w
            self.B = self.B - step*delta_b
            if idx%10 == 0:
                print( 'Finished step ',idx,' loss = ',entropy_loss/N)
        
    # -- - - Training 
    def train( self, dataset, labels ):
        #
        self.minimize( dataset, labels )

    # -- - - Prediction
    def predict( self, X ):
        temp = np.dot( self.W, X ) + self.B
        return np.argmax( temp )
