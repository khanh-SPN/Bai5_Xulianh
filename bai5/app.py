import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading
from naive_bayes_classifier import train_naive_bayes
from cart_classifier import train_cart
from id3_classifier import train_id3
from neural_network import build_neural_network
from data_preparation import load_dental_images
from evaluation import evaluate_model

# Tạo cửa sổ chính
app = tk.Tk()
app.title("Ứng dụng Phân Lớp Ảnh")
app.geometry("500x500")

# Thêm nhãn để hiển thị trạng thái
status_label = tk.Label(app, text="Chọn một thuật toán để bắt đầu phân lớp.", font=("Arial", 12))
status_label.pack(pady=10)

# Thêm khung cuộn để hiển thị báo cáo phân lớp chi tiết
result_text = scrolledtext.ScrolledText(app, wrap=tk.WORD, width=60, height=20, font=("Arial", 10))
result_text.pack(pady=10)

# Hàm hiển thị thông báo kết quả và báo cáo chi tiết
def show_result(model_name, accuracy, report):
    status_label.config(text=f"{model_name} - Độ chính xác: {accuracy:.2f}")
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"=== {model_name} ===\nĐộ chính xác: {accuracy:.2f}\n\nBáo cáo phân lớp:\n{report}\n")

# Hàm chạy mô hình trong một luồng riêng
def run_algorithm(algorithm_func, model_name):
    status_label.config(text=f"Đang chạy {model_name}... Vui lòng chờ.")
    app.update_idletasks()  # Cập nhật giao diện ngay lập tức

    # Tải dữ liệu ảnh cho mỗi mô hình khi chạy
    X_train, X_test, y_train, y_test = load_dental_images()
    
    # Khởi chạy mô hình và lấy kết quả
    model, accuracy, report = algorithm_func(X_train, X_test, y_train, y_test)
    
    # Hiển thị kết quả
    show_result(model_name, accuracy, report)

# Các hàm cho từng thuật toán
def run_naive_bayes():
    threading.Thread(target=run_algorithm, args=(train_naive_bayes_dental, "Naive Bayes")).start()

def run_cart():
    threading.Thread(target=run_algorithm, args=(train_cart_dental, "CART (Gini Index)")).start()

def run_id3():
    threading.Thread(target=run_algorithm, args=(train_id3_dental, "ID3 (Information Gain)")).start()

def run_neural_network():
    threading.Thread(target=run_algorithm, args=(train_neural_network_dental, "Neural Network")).start()

# Các hàm huấn luyện cho từng thuật toán, trả về độ chính xác và báo cáo phân lớp
def train_naive_bayes_dental(X_train, X_test, y_train, y_test):
    model = train_naive_bayes(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test, "Naive Bayes - Dental Images")
    return model, accuracy, report

def train_cart_dental(X_train, X_test, y_train, y_test):
    model = train_cart(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test, "CART - Dental Images")
    return model, accuracy, report

def train_id3_dental(X_train, X_test, y_train, y_test):
    model = train_id3(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test, "ID3 - Dental Images")
    return model, accuracy, report

def train_neural_network_dental(X_train, X_test, y_train, y_test):
    model = build_neural_network((X_train.shape[1],), len(set(y_train)))
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
    accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
    report = "Báo cáo phân lớp cho Neural Network không sẵn có trong định dạng chi tiết."  # Báo cáo đơn giản
    return model, accuracy, report

# Thêm các nút cho từng thuật toán
btn_naive_bayes = tk.Button(app, text="Naive Bayes", command=run_naive_bayes, font=("Arial", 12), width=20)
btn_naive_bayes.pack(pady=5)

btn_cart = tk.Button(app, text="CART (Gini Index)", command=run_cart, font=("Arial", 12), width=20)
btn_cart.pack(pady=5)

btn_id3 = tk.Button(app, text="ID3 (Information Gain)", command=run_id3, font=("Arial", 12), width=20)
btn_id3.pack(pady=5)

btn_neural_network = tk.Button(app, text="Neural Network", command=run_neural_network, font=("Arial", 12), width=20)
btn_neural_network.pack(pady=5)

# Chạy ứng dụng
app.mainloop()
