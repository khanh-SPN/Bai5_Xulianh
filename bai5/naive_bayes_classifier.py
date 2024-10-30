from sklearn.naive_bayes import GaussianNB
from data_preparation import load_iris_data, load_dental_images
from evaluation import evaluate_model
from sklearn.preprocessing import StandardScaler

def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

# Chuẩn hóa và huấn luyện cho bộ dữ liệu IRIS
X_train_iris, X_test_iris, y_train_iris, y_test_iris = load_iris_data()

# Chuẩn hóa dữ liệu Iris
scaler_iris = StandardScaler()
X_train_iris = scaler_iris.fit_transform(X_train_iris)
X_test_iris = scaler_iris.transform(X_test_iris)

# Huấn luyện mô hình Naive Bayes cho Iris và đánh giá
model_iris = train_naive_bayes(X_train_iris, y_train_iris)
accuracy_iris, report_iris = evaluate_model(model_iris, X_test_iris, y_test_iris, "Naive Bayes - IRIS")
print(f"Độ chính xác Naive Bayes - IRIS: {accuracy_iris:.2f}")
print(report_iris)

# Chuẩn hóa và huấn luyện cho bộ dữ liệu ảnh nha khoa
X_train_dental, X_test_dental, y_train_dental, y_test_dental = load_dental_images()

# Chuẩn hóa dữ liệu ảnh nha khoa
scaler_dental = StandardScaler()
X_train_dental = scaler_dental.fit_transform(X_train_dental)
X_test_dental = scaler_dental.transform(X_test_dental)

# Huấn luyện mô hình Naive Bayes cho ảnh nha khoa và đánh giá
model_dental = train_naive_bayes(X_train_dental, y_train_dental)
accuracy_dental, report_dental = evaluate_model(model_dental, X_test_dental, y_test_dental, "Naive Bayes - Dental Images")
print(f"Độ chính xác Naive Bayes - Dental Images: {accuracy_dental:.2f}")
print(report_dental)
