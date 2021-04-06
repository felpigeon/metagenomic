from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np

xTrain = np.load('xTrain.npy')
yTrain = np.load('yTrain.npy')

kernels = ['linear', 'poly2', 'poly', 'rbf', 'sigmoid']
c_array = [0.01, 0.1, 1, 10, 100]
tableau = []

for i in kernels:
    for j in c_array:
        print("i= " + str(i) + ", j= " + str(j))
        if i == 'poly2':
            svm = SVC(kernel='poly', degree=2, C=j, class_weight='balanced', probability=True)
        else:
            svm = SVC(kernel=i, C=j, class_weight='balanced', probability=True)

        scores = cross_val_score(svm, xTrain, yTrain, cv=5, scoring='roc_auc')
        scores_mean = np.mean(scores)

        print(scores_mean)
        tableau.append([scores_mean, i, j])

print(tableau)
