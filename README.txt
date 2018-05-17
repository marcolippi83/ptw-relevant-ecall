Machine Learning for Severity Prediction of Accidents Involving Powered Two Wheelers

This repository contains the following folders:

* data
  a CSV file containing the dataset used in our experiments

* scripts
  python and bash scripts to run three different classifiers
  (Random Forests, Deep Networks, Support Vector Machines)

* results
  the results presented in our paper, obtained on 20-fold cross validation
  both in labels and predictions: 1=relevant, 0=non-relevant

To run the code, please edit the scripts/run.sh file by changing the PYTHON variable accordingly.
The code requires the following python libraries:

* keras
* tensorflow
* scikit-learn
* numpy
* hyperas



