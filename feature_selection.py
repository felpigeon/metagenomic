import numpy as np

## Création de données
x1 = np.random.random((10000, 100))-0.5
x2 = np.random.random((10000, 300))-0.5

x = np.concatenate([x1,x2], axis=1)

y = x1.sum(axis=-1)
y = y > 0

print(x.shape, y.shape)

## Split
xTrain, yTrain = x[:9000], y[:9000]
xTest, yTest = x[9000:], y[9000:]


## Feature selection
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(xTrain, yTrain)

mean_score = 1 / x.shape[1]
importances = rf.feature_importances_
print('Mean score', mean_score)
print('Feature importances', importances)


#Méthode 1
features = np.arange(x.shape[1])[ importances > mean_score ]
print(features)

#Méthode 2
features = np.argsort( importances )
print(features[ -int(x.shape[1]/10):] )


