# *******************************************************************************
# **
# **  Author: Michael Lomnitz (mrlomnitz@lbl.gov)                               
# **  Python module running neural network with one hidden layer with 1024 
# **  neurons, dropout and relu activation.
# **
# *******************************************************************************
# Import relevant modules
import numpy as np
import tensorflow as tf

class nn(object):
    def __init__(self,layers, image_size, num_labels, nodes):
        assert layers-1 == len(nodes)
        self.layers = layers
        _matrix_dims = [image_size * image_size] + nodes + [num_labels]
        self.weights = []
        self.biases = []
        print(_matrix_dims)
        for i_layer in range(0,layers):
            self.weights.append( 
                tf.Variable(tf.truncated_normal([_matrix_dims[i_layer],_matrix_dims[i_layer+1]])))
            self.biases.append(tf.Variable(tf.zeros([ _matrix_dims[i_layer+1] ])))
        
    def train(self, x, y, keep_prob):
        logits = x
        for i_layer in range(0,self.layers):
            logits = tf.matmul(logits, self.weights[i_layer]) + self.biases[i_layer]
            #relu on hidden layers
            if i_layer != self.layers -1 :
                logits = tf.nn.relu(logits)
                logits = tf.nn.dropout(logits,keep_prob)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
        # L2 Loss
        for i_layer in range(0,self.layers):
            loss+= 0.01*tf.nn.l2_loss(self.weights[i_layer])
        #
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.1,global_step, 500, 0.9)
        #
        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss, global_step=global_step)
        return logits, loss, optimizer
    
    def predict(self,x):
        logits = x
        for i_layer in range(0,self.layers):
            logits = tf.matmul(logits, self.weights[i_layer]) + self.biases[i_layer]
            #relu on hidden layers
            if i_layer != self.layers -1 :
                logits = tf.nn.relu(logits)
    
        return tf.nn.softmax( logits )
