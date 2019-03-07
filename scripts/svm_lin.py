import sys
import pandas as pd
import numpy as np
import scipy
from pandas import Series
from math import sqrt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
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

# Labels
Y = df["Severity"]
# Features
X = df.drop(['Severity'], axis=1).values


def stand_data(X_train,X_test):
    # Standardize data (0 mean, 1 stdev)
    scaler = StandardScaler().fit(X_train)
    s_X_train = scaler.transform(X_train)
    s_X_test = scaler.transform(X_test)
    return s_X_train, s_X_test

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

    tuned_parameters = [{'kernel': ['linear'], 'C': [0.01,0.1,1,10,100]}]

    if (cont == 1):
        clf = GridSearchCV(SVC(class_weight='balanced'), tuned_parameters, cv=3, scoring='f1_macro')
        clf.fit(X_train,Y_train)
        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()
        clf_best = clf.best_estimator_

    clf = clf_best
    clf.fit(X_train,Y_train)
    y_pred = clf.predict(X_test)

    predictions = clf.predict(X_test)
    y_true = Y_test
    y_pred = predictions
    np.savetxt('pred_svm_lin_' + str(cont) + ".txt", y_pred)
    np.savetxt('true_svm_lin_' + str(cont) + ".txt", y_true)
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
