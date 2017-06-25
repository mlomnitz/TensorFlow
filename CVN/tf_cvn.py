# *******************************************************************************
# **
# **  Author: Michael Lomnitz (mrlomnitz@lbl.gov)                               
# **  Python module defining Convolutional Neural network model with one hidden
# **  hidden layer, relu activation, same padding & maxPool for stride. Dropout 
# **  included to avoid overtraining, minibatches implemented to speed up traning 
# **  and option for several epochs included.
# **
# *******************************************************************************
# Import relevant modules
import numpy as np
import tensorflow as tf

# two conv layers and a fully connected. 
def maxPool2D(x,k=2):
    return tf.nn.max_pool(x, ksize = [1,k,k,1], strides = [1,2,2,1],padding = 'SAME')

class cvn(object):

    #
    def __init__(self, image_size, batch_size, patch_size, num_channels, depth, num_hidden, num_labels):
        # Variables.
        self.layer1_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, num_channels, depth], stddev=0.1))
        self.layer1_biases = tf.Variable(tf.zeros([depth]))
        self.layer2_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth, depth], stddev=0.1))
        self.layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
        self.layer3_weights = tf.Variable(tf.truncated_normal(
          [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
        self.layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
        self.layer4_weights = tf.Variable(tf.truncated_normal(
          [num_hidden, num_labels], stddev=0.1))
        self.layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))   
        
    def model(self, data, keep_prob, stride = 1):
        #. Two convolutions, max pooling k = 2, stride = 2
        conv = tf.nn.conv2d(data, self.layer1_weights, [1, stride, stride, 1], padding='SAME')        
        conv = maxPool2D(conv, k=2)
        hidden = tf.nn.relu(conv + self.layer1_biases)
        conv = tf.nn.conv2d(hidden, self.layer2_weights, [1, stride, stride, 1], padding='SAME')
        conv = maxPool2D(conv, k=2)
        hidden = tf.nn.relu(conv + self.layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        #Fully connected layer
        hidden = tf.nn.relu(tf.matmul(reshape, self.layer3_weights) + self.layer3_biases)
        hidden = tf.nn.dropout(hidden, keep_prob)
        return tf.matmul(hidden, self.layer4_weights) + self.layer4_biases
        
    
    def train(self, x, y, keep_prob):
        print(x.shape)
        logits = self.model(x, keep_prob)
        #  - - -Loss
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
        #  - - - Optimizer and learning arate
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.1,global_step, 500, 0.9)
        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss, global_step=global_step)
        return logits, loss, optimizer
    
    def predict(self, x, y):
        print(x.shape)
        logits = self.model(x, 1)
        #  - - -Loss
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
        return logits, loss

