import pandas as pd
import numpy as np
import scipy
import sys
from pandas import Series
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import learning_curve
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from numpy.random import seed
seed(1234)
from tensorflow import set_random_seed
set_random_seed(4567)

def stand_data(X_train,X_test):
    # Standardize data (0 mean, 1 stdev)
    scaler = StandardScaler().fit(X_train)
    s_X_train = scaler.transform(X_train)
    s_X_test = scaler.transform(X_test)
    return s_X_train, s_X_test

def data(X_train,Y_train,X_test,Y_test):
    X_train = np.loadtxt('tmp_X_train.txt')
    X_test = np.loadtxt('tmp_X_test.txt')
    Y_train = np.loadtxt('tmp_y_train.txt')
    Y_test = np.loadtxt('tmp_y_test.txt')

    return X_train, Y_train, X_test, Y_test

def create_model(X_train, Y_train, X_test, Y_test):
    pos = len(Y_train[Y_train == 1])
    neg = len(Y_train) - pos
    neg_pos_ratio = float(neg)/float(pos)
    class_weight = {0:1.0, 1:neg_pos_ratio}

    model = Sequential()
    model.add(Dense({{choice([128, 256, 512])}}, input_dim=31,activation ='relu'))  #hidden layer
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([64, 128, 256])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([32, 64, 128])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([16, 32, 64])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([8, 16, 32])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(1, activation='sigmoid'))  #output layer

    adam = Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_acc', patience=100, verbose=0, mode='auto')
    model.fit(X_train, Y_train, epochs=100000, verbose=2, batch_size=100,callbacks=[early_stopping], validation_split=0.05, class_weight=class_weight)
    model.evaluate(X_test, Y_test, verbose=2)
    predictions = model.predict(X_test)
    y_true = Y_test
    y_pred = [round(x[0]) for x in predictions]
    acc = accuracy_score(y_true,y_pred)
    pre = precision_score(y_true,y_pred)
    rec = recall_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred)

    return {'loss': -f1, 'status': STATUS_OK, 'model': model}


df=pd.read_csv(sys.argv[1],delimiter = ';',skiprows = 0)
df=df.dropna(axis=0,how='any')
df= df.values

# Features
X = df[:,1:32]
# Labels
Y = df[:,0]

np.set_printoptions(precision=3)

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
    s_X_train,s_X_test = stand_data(X_train,X_test)
    np.savetxt('tmp_X_train.txt',s_X_train)
    np.savetxt('tmp_X_test.txt',s_X_test)
    np.savetxt('tmp_y_train.txt',Y_train)
    np.savetxt('tmp_y_test.txt',Y_test)

    best_run, best_model = optim.minimize(model=create_model, data=data, algo=tpe.suggest, max_evals=5, trials=Trials())
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    predictions = best_model.predict(s_X_test)
    y_true = Y_test
    y_pred = [round(x[0]) for x in predictions]
    np.savetxt('pred_dnn_' + str(cont) + ".txt", y_pred)
    np.savetxt('true_dnn_' + str(cont) + ".txt", y_true)
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

    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    cont = cont + 1
    accuracy.append(acc)
    precision.append(pre)
    recall.append(rec)
    fmeasure.append(f1)

m_acc = np.mean(accuracy)
m_pre = np.mean(precision)
m_rec = np.mean(recall)
m_f1 = np.mean(fmeasure)
print('Mean Accuracy = ' + str(m_acc))
print('Mean Precision = ' + str(m_pre))
print('Mean Recall = ' + str(m_rec))
print('Mean F1 = ' + str(m_f1))
