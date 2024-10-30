from sklearn.naive_bayes import GaussianNB
from data_preparation import load_iris_data, load_dental_images
from evaluation import evaluate_model

def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

# IRIS dataset
X_train_iris, X_test_iris, y_train_iris, y_test_iris = load_iris_data()
model_iris = train_naive_bayes(X_train_iris, y_train_iris)
evaluate_model(model_iris, X_test_iris, y_test_iris, "Naive Bayes - IRIS")

# Dental images dataset
X_train_dental, X_test_dental, y_train_dental, y_test_dental = load_dental_images()
model_dental = train_naive_bayes(X_train_dental, y_train_dental)
evaluate_model(model_dental, X_test_dental, y_test_dental, "Naive Bayes - Dental Images")
