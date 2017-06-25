# *******************************************************************************
# **
# **  Author: Michael Lomnitz (mrlomnitz@lbl.gov)                               
# **  Python module running convolutional neural network 
# **
# *******************************************************************************
import sys
sys.path.append('../notMNIST_dataset')
import tensorflow as tf
import numpy as np
# Local modules
import tf_cvn as cvn
import Load_Dataset as dl

# Probvlem constants
image_size = 28
num_labels = 10
num_channels = 1 # grayscale
patch_size = 5
depth = 16
num_hidden = 64
#
num_epochs = 5
training_samples = 100000
batch_size =40
num_steps = int(training_samples/batch_size)+1
drop_out_prob = 0.75
#
data =  dl.data_set(training_samples, True)
graph = tf.Graph()

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with graph.as_default():    
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    
      # Input data.
    tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(data.valid_dataset)
    tf_test_dataset = tf.constant(data.test_dataset)
    keep_prob = tf.placeholder(tf.float32)
    nodes = [1024]
    conv = cvn.cvn(image_size, batch_size, patch_size, num_channels, depth, num_hidden, num_labels)
    # Training computation.
    logits, loss, optimizer = conv.train(tf_train_dataset, tf_train_labels, keep_prob)
  
    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(conv.model(tf_valid_dataset,1))
    test_prediction = tf.nn.softmax(conv.model(tf_test_dataset,1))

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for epoch in range(0,num_epochs):
        for step in range(num_steps):
            offset = (step * batch_size) % (data.train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = data.train_dataset[offset:(offset + batch_size), :]
            batch_labels = data.train_labels[offset:(offset + batch_size), :]
            #
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : drop_out_prob}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 100 == 0):
                print('Minibatch loss epoch %d at step %d: %f' % (epoch,step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
   
        print("End of epoch ",epoch)
        print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), data.valid_labels))
