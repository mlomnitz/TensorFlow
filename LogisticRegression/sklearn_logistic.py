# Import relevant modules
import sys
sys.path.append('../notMNIST_dataset')
from sklearn.linear_model import LogisticRegression
import Load_Dataset as dl
import numpy as np

def get_real_label( labelset ):
    nx,ny = labelset.shape
    to_ret = np.argmax( labelset, axis = 1 )
    return to_ret

def train( dataset, labelset ):
    logistic = LogisticRegression()
    logistic.fit( dataset,get_real_label(labelset) )
    return logistic

def test( dataset, labelset ):
    failed = 0
    predictions = logistic.predict(dataset)
    test_labels = get_real_label( labelset )
    n_samples = predictions.shape
    print( n_samples, len(predictions) )
    for idx,pred in enumerate(predictions):
        if pred != test_labels[idx]:
            failed+=1
    print("Test samplee accuracy = ",int(100-failed/len(predictions)*100),"% ")    

index_to_letter = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J'}
data = dl.data_set(1000 )
logistic = train( data.train_dataset, data.train_labels )
test( data.test_dataset, data.test_labels )

