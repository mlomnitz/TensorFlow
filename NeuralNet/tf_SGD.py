# *******************************************************************************
# **
# **  Author: Michael Lomnitz (mrlomnitz@lbl.gov)                               
# **  Python module defining stochastic gradient descent model for use in 
# **  tensorflow classifier
# **
# *******************************************************************************
# Import relevant modules
import numpy as np
import tensorflow as tf

class SGD(object):
    def __init__(self,image_size,num_labels):
        self.weights = tf.Variable(
            tf.truncated_normal([image_size * image_size, num_labels]))
        self.biases = tf.Variable(tf.zeros([num_labels]))
        #
    def train(self,x, y):
        logits = tf.matmul(x, self.weights) + self.biases
        loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))  
        # Optimizer.
        # We are going to find the minimum of this loss using gradient descent.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        return logits, loss, optimizer

    def predict(self,x):
        return tf.nn.softmax( tf.matmul(x,self.weights) + self.biases)  

