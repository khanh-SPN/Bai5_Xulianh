from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, X_test, y_test, model_name=""):
    predictions = model.predict(X_test)
    print(f"Đánh giá mô hình: {model_name}")
    print("Báo cáo phân lớp:\n", classification_report(y_test, predictions, zero_division=1))
    accuracy = accuracy_score(y_test, predictions)
    print("Độ chính xác:", accuracy)
    return accuracy
