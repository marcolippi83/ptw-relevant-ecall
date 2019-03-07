PYTHON=python3

dataset=dataset_A.csv
#dataset=dataset_B.csv
#dataset=dataset_C.csv
#dataset=dataset_D.csv

$PYTHON lr.py ../data/$dataset > RESULTS_LR.txt
$PYTHON rf.py ../data/$dataset > RESULTS_RF.txt
$PYTHON svm_lin.py ../data/$dataset > RESULTS_SVM_LIN.txt
$PYTHON svm_rbf.py ../data/$dataset > RESULTS_SVM_RBF.txt
$PYTHON dnn.py ../data/$dataset > RESULTS_DNN.txt

