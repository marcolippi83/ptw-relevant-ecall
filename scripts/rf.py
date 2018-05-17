import sys
import pandas as pd
import numpy as np
import scipy
from pandas import Series
from math import sqrt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from numpy.random import seed
seed(1234)
from tensorflow import set_random_seed
set_random_seed(4567)

df=pd.read_csv(sys.argv[1],delimiter = ';',skiprows = 0)
df=df.drop(['id'], axis=1)
df=df.dropna(axis=0,how='any')
df=df.values

# Features
X = df[:,1:]
# Labels
Y = df[:,0]


skf = StratifiedKFold(n_splits=20)

accuracy = []
precision = []
recall = []
fmeasure = []
cont=1
for train_index, test_index in skf.split(X, Y):
    print(cont,"TRAIN:", train_index, cont,"TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    clf = RandomForestClassifier(class_weight="balanced", n_estimators=1000)
    clf.fit(X_train,Y_train)
    y_pred = clf.predict(X_test)
    importances = clf.feature_importances_
    #print(importances)

    predictions = clf.predict(X_test)
    y_true = Y_test
    y_pred = predictions
    np.savetxt('pred_rf_' + str(cont) + ".txt", y_pred, fmt='%d')
    np.savetxt('true_rf_' + str(cont) + ".txt", y_true, fmt='%d')
    c_matrix = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true,y_pred)
    pre = precision_score(y_true,y_pred)
    rec = recall_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred)
    print('Confusion_matrix ',cont,'Test:')
    print(c_matrix)
    print('Accuracy ',cont,'Test:')
    print(acc)
    print('Precision ',cont,'Test:')
    print(pre)
    print('Recall ',cont,'Test:')
    print(rec)
    print('F1 ',cont,'Test:')
    print(f1)
    print()
    cont = cont + 1
    accuracy.append(acc)
    precision.append(pre)
    recall.append(rec)
    fmeasure.append(f1)

print('Accuracy')
print(accuracy)
print('Precision')
print(precision)
print('Recall')
print(recall)
print('F1')
print(fmeasure)

m_acc = np.mean(accuracy)
m_pre = np.mean(precision)
m_rec = np.mean(recall)
m_f1 = np.mean(fmeasure)
print('Mean Accuracy = ' + str(m_acc))
print('Mean Precision = ' + str(m_pre))
print('Mean Recall = ' + str(m_rec))
print('Mean F1 = ' + str(m_f1))
