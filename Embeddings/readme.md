# Word embeddings
This directory contains the implementation determining a words "meaning" from the contextial information (word embeddings). Two models are included to generate a 
supervised learning algorithm from a corpus:
1. Skipgram: Trains the model to learn to predict the context given the appearance of a words.
2. Continuous bag of words (CBOW): To some degree inverse of skipgram, trains the model to learn the meaning of a words given the context (adjacent words) in which it appears.

This directory contains the follwing files:

* _Load_Text_Set.py_ : Loads the dataset (corpus) used for training and validation. Downlaods dataset provided by tensorflow and loads the .zip into a usable dictionary.
* _tf_Word2Vec.py_ : Defines the word2vec models (skipgram or cbow) and necesary methods to feed them the embeddings data.
* _run_Word2Vec.py_ : Script that utilizes theprevious two modules to load the data, feed it to embeddings models and do the training. A random validation set is generated from the data used to test performance. Performance is visualized using t-SNE implementation from SKlearn.

The following illustrate the performance of the two models:
1. Skipgram:
![skipgram_model](https://github.com/mlomnitz/TensorFlow/blob/master/Embeddings/Embeddings.pdf = 150x150)
2. CBOW:
![CBOW_model](https://github.com/mlomnitz/TensorFlow/blob/master/Embeddings/Embeddings_CBOW.pdf  = 150x150)
