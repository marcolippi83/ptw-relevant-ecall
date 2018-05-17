PYTHON=python3.6

$PYTHON rf.py ../data/ptw.db > RESULTS_RF.txt
$PYTHON svm.py ../data/ptw.db > RESULTS_SVM.txt
$PYTHON dnn.py ../data/ptw.db > RESULTS_DNN.txt

