from data_preparation import load_x_and_y

#Chargement de toutes les features
xTest, yTest = load_x_and_y( 'data/test_idx.npy' )
print( 'Testing set shapes', xTest.shape, yTest.shape)

xTrain, yTrain = load_x_and_y( 'data/train_idx.npy' )
print( 'Training set shapes', xTrain.shape, yTrain.shape)


#Chargement des features sélectionnées
xTest, yTest = load_x_and_y( 'data/test_idx.npy', 'data/features_idx.npy' )
print( 'Testing set shapes', xTest.shape, yTest.shape)

xTrain, yTrain = load_x_and_y( 'data/train_idx.npy', 'data/features_idx.npy' )
print( 'Training set shapes', xTrain.shape, yTrain.shape)
