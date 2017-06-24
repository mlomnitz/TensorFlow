# *******************************************************************************
# **
# **  Author: Michael Lomnitz (mrlomnitz@lbl.gov)                               
# **  Python module loading stochastic gradient descent model defined in tf_SGD.py
# **  for classification.
# **
# *******************************************************************************
import sys
sys.path.append('../notMNIST_dataset')
import tensorflow as tf
import numpy as np
# Local modules
import tf_SGD as model
import Load_Dataset as dl

data =  dl.data_set(1000)
batch_size = 128
num_steps = 3001
graph = tf.Graph()
image_size = 28
num_labels = 10

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(data.valid_dataset)
    tf_test_dataset = tf.constant(data.test_dataset)
    SGD = model.SGD(image_size, num_labels)
    # Training computation.
    logits, loss, optimizer = SGD.train(tf_train_dataset, tf_train_labels)
    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = SGD.predict(tf_valid_dataset)
    test_prediction = SGD.predict(tf_test_dataset)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (data.train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = data.train_dataset[offset:(offset + batch_size), :]
        batch_labels = data.train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), data.valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), data.test_labels))
