# *******************************************************************************
# **
# **  Author: Michael Lomnitz (mrlomnitz@lbl.gov)                               
# **  Python module used to load pickled dictionary of images and associated 
# **  labels for classification.
# **
# *******************************************************************************
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = '/Users/michaellomnitz/Documents/Jupyter/DeepLearning/TensorFlow/notMNIST_dataset/notMNIST.pickle'
image_size = 28
num_labels = 10

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

 
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

def overlap( dataset_1, dataset_2):
    entries_1, pixx_1,pixy_1 = dataset_1.shape
    entries_2, pixx_2,pixy_2 = dataset_2.shape
    my_dict = {}
    a = {}
    to_ret = []
    for idx, x in enumerate(dataset_1):
        #print (x)
        my_dict.update({str(x):idx})
    
    print("Done loading first dataset")  
    
    for idy, y in enumerate(dataset_2):
        if str(y) in my_dict:
            a.update({my_dict[str(y)]:idy})
        else:
            to_ret.append(y)
            
    print(len(a),"shared dictionaries")
    print(a)
    return to_ret


class data_set(object):
    
    def __init__(self, n_samples = 0):

        with open(pickle_file, 'rb') as f:
            save = pickle.load(f)
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            valid_dataset = save['valid_dataset']
            valid_labels = save['valid_labels']
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
            del save  # hint to help gc free up memory
            print('Training set', train_dataset.shape, train_labels.shape)
            print('Validation set', valid_dataset.shape, valid_labels.shape)
            print('Test set', test_dataset.shape, test_labels.shape)
        self.train_dataset, self.train_labels = reformat(train_dataset, train_labels)
        self.valid_dataset, self.valid_labels = reformat(valid_dataset, valid_labels)
        self.test_dataset, self.test_labels = reformat(test_dataset, test_labels)

        if n_samples != 0:
            self.train_dataset = self.train_dataset[:n_samples , : ]
            self.train_labels = self.train_labels[:n_samples , : ] 
        
        print('Training set', self.train_dataset.shape, self.train_labels.shape)
        print('Validation set', self.valid_dataset.shape, self.valid_labels.shape)
        print('Test set', self.test_dataset.shape, self.test_labels.shape)
        
        return 

