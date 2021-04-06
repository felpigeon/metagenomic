from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np

xTrain = np.load('xTrain.npy')
yTrain = np.load('yTrain.npy')

n_estimators = [10, 100, 1000]
impurity = [0, 10**(-3), 10**(-6)]
max_features = [0.1, 'sqrt', None]
tableau = []


for i in n_estimators:
    for j in impurity:
        for k in max_features:
            rf = RandomForestClassifier(n_estimators=i, min_impurity_decrease=j, max_features=k,
                                        class_weight='balanced')
            scores = cross_val_score(rf, xTrain, yTrain, cv=5, scoring='roc_auc')
            scores_mean = np.mean(scores)
            tableau.append([scores_mean, i, j, k])

print(tableau)
