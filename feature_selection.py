import numpy as np
from sklearn.ensemble import RandomForestClassifier

from data_preparation import load_x_and_y

xTrain, yTrain = load_x_and_y('data/train_idx.npy')
print('Training set shapes', xTrain.shape, yTrain.shape)


# Feature selection
rf = RandomForestClassifier()
rf.fit(xTrain, yTrain)

mean_score = 1 / xTrain.shape[1]
importances = rf.feature_importances_
print('Mean score', mean_score)
print('Feature importances', importances)

# Top 200
features = np.argsort(importances)[-200:]

# print(features[features > 288347])
print(len(features[features > 288347]))  # ~ 650

np.save('data/features_idx.npy', features)
