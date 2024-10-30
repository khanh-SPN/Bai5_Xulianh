from sklearn.tree import DecisionTreeClassifier
from data_preparation import load_iris_data, load_dental_images
from evaluation import evaluate_model

def train_cart(X_train, y_train):
    model = DecisionTreeClassifier(criterion='gini')
    model.fit(X_train, y_train)
    return model

# IRIS dataset
X_train_iris, X_test_iris, y_train_iris, y_test_iris = load_iris_data()
model_iris = train_cart(X_train_iris, y_train_iris)
evaluate_model(model_iris, X_test_iris, y_test_iris, "CART - IRIS")

# Dental images dataset
X_train_dental, X_test_dental, y_train_dental, y_test_dental = load_dental_images()
model_dental = train_cart(X_train_dental, y_train_dental)
evaluate_model(model_dental, X_test_dental, y_test_dental, "CART - Dental Images")
