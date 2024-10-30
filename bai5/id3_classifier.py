from sklearn.tree import DecisionTreeClassifier
from data_preparation import load_iris_data, load_dental_images
from evaluation import evaluate_model

def train_id3(X_train, y_train):
    # Tạo mô hình DecisionTree với siêu tham số được tối ưu hóa
    model = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=5,               # Giới hạn độ sâu của cây
        min_samples_split=10,      # Số lượng mẫu tối thiểu để chia một nút
        min_samples_leaf=5         # Số lượng mẫu tối thiểu tại một lá
    )
    model.fit(X_train, y_train)
    return model

# Dataset IRIS
X_train_iris, X_test_iris, y_train_iris, y_test_iris = load_iris_data()
model_iris = train_id3(X_train_iris, y_train_iris)
accuracy_iris, report_iris = evaluate_model(model_iris, X_test_iris, y_test_iris, "ID3 - IRIS")
print(f"Độ chính xác ID3 - IRIS: {accuracy_iris:.2f}")
print(report_iris)

# Dataset ảnh nha khoa
X_train_dental, X_test_dental, y_train_dental, y_test_dental = load_dental_images()
model_dental = train_id3(X_train_dental, y_train_dental)
accuracy_dental, report_dental = evaluate_model(model_dental, X_test_dental, y_test_dental, "ID3 - Dental Images")
print(f"Độ chính xác ID3 - Dental Images: {accuracy_dental:.2f}")
print(report_dental)
