from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, X_test, y_test, model_name=""):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, zero_division=1)
    accuracy = accuracy_score(y_test, predictions)
    
    # Hiển thị báo cáo chi tiết
    print(f"Đánh giá mô hình: {model_name}")
    print("Báo cáo phân lớp:\n", report)
    print("Độ chính xác:", accuracy)
    
    # Trả về cả accuracy và report để sử dụng sau này
    return accuracy, report
