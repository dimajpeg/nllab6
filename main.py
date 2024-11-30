# main.py
from models.svm_model import train_svm_model
from models.knn_model import train_knn_model
from evaluation.evaluate import evaluate_model

csv_path = 'data/dataset.csv'

# Обучение и оценка модели SVM
svm_model, X_test_svm, y_test_svm = train_svm_model(csv_path)
print("Evaluating SVM Model...")
evaluate_model(svm_model, X_test_svm, y_test_svm)

# Обучение и оценка модели KNN
knn_model, X_test_knn, y_test_knn = train_knn_model(csv_path)
print("Evaluating KNN Model...")
evaluate_model(knn_model, X_test_knn, y_test_knn)
