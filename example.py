import numpy as np

from data import load_x_and_y

xTest, yTest = load_x_and_y( 'test_idx.npy' )
print( 'Testing set shapes', xTest.shape, yTest.shape)

xTrain, yTrain = load_x_and_y( 'train_idx.npy' )
print( 'Training set shapes', xTrain.shape, yTrain.shape)


