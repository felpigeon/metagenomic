import numpy as np
from data import load_x_and_y

xTest, yTest = load_x_and_y('test_idx.npy')
print('Testing set shapes', xTest.shape, yTest.shape)

xTrain, yTrain = load_x_and_y('train_idx.npy')
print('Training set shapes', xTrain.shape, yTrain.shape)

# print("x: " + str(xTrain))
# print("y: " + str(yTrain))
# print("test: " + str(np.count_nonzero(xTrain)))


# Feature selection
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(xTrain, yTrain)

mean_score = 1 / xTrain.shape[1]
importances = rf.feature_importances_
print('Mean score', mean_score)
print('Feature importances', importances)

# Méthode 1
features = np.arange(xTrain.shape[1])[importances > mean_score]
# print(features)
print(len(features))  # ~ 2200-2300

# print(features[features > 288347])
print(len(features[features > 288347]))  # ~ 650
