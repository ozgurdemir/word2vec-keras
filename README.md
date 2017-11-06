# Word2Vec in Keras

This is a simple word2vec implementation with negative sampling in Keras.

## How to run it

    > make example
    
Will download an example text file (the one used in the original documentation) and start the training
process in a docker container. 

    > make predict WORD=king
    
Will use the trained model to predict words similar to the entered word. 