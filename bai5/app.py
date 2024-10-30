import tkinter as tk
from tkinter import messagebox
import threading
from naive_bayes_classifier import train_naive_bayes
from cart_classifier import train_cart
from id3_classifier import train_id3
from neural_network import build_neural_network
from data_preparation import load_dental_images
from evaluation import evaluate_model

# Tải dữ liệu ảnh
X_train_dental, X_test_dental, y_train_dental, y_test_dental = load_dental_images()

# Tạo cửa sổ chính
app = tk.Tk()
app.title("Ứng dụng Phân Lớp Ảnh")
app.geometry("400x350")

# Thêm nhãn để hiển thị trạng thái
status_label = tk.Label(app, text="Chọn một thuật toán để bắt đầu phân lớp.", font=("Arial", 12))
status_label.pack(pady=20)

# Hàm hiển thị thông báo kết quả
def show_result(model_name, accuracy):
    status_label.config(text=f"{model_name} - Độ chính xác: {accuracy:.2f}")
    messagebox.showinfo("Kết quả", f"{model_name} - Độ chính xác: {accuracy:.2f}")

# Hàm chạy mô hình trong một luồng riêng
def run_algorithm(algorithm_func, model_name):
    status_label.config(text=f"Đang chạy {model_name}... Vui lòng chờ.")
    app.update_idletasks()  # Cập nhật giao diện ngay lập tức
    
    # Khởi chạy mô hình
    model, accuracy = algorithm_func()
    
    # Hiển thị kết quả
    show_result(model_name, accuracy)

# Các hàm cho từng thuật toán
def run_naive_bayes():
    threading.Thread(target=run_algorithm, args=(train_naive_bayes_dental, "Naive Bayes")).start()

def run_cart():
    threading.Thread(target=run_algorithm, args=(train_cart_dental, "CART (Gini Index)")).start()

def run_id3():
    threading.Thread(target=run_algorithm, args=(train_id3_dental, "ID3 (Information Gain)")).start()

def run_neural_network():
    threading.Thread(target=run_algorithm, args=(train_neural_network_dental, "Neural Network")).start()

# Các hàm huấn luyện cho từng thuật toán, trả về độ chính xác
def train_naive_bayes_dental():
    model = train_naive_bayes(X_train_dental, y_train_dental)
    accuracy = evaluate_model(model, X_test_dental, y_test_dental, "Naive Bayes - Dental Images")
    return model, accuracy

def train_cart_dental():
    model = train_cart(X_train_dental, y_train_dental)
    accuracy = evaluate_model(model, X_test_dental, y_test_dental, "CART - Dental Images")
    return model, accuracy

def train_id3_dental():
    model = train_id3(X_train_dental, y_train_dental)
    accuracy = evaluate_model(model, X_test_dental, y_test_dental, "ID3 - Dental Images")
    return model, accuracy

def train_neural_network_dental():
    model = build_neural_network((X_train_dental.shape[1],), len(set(y_train_dental)))
    model.fit(X_train_dental, y_train_dental, epochs=10, batch_size=16, validation_data=(X_test_dental, y_test_dental))
    accuracy = model.evaluate(X_test_dental, y_test_dental, verbose=0)[1]
    return model, accuracy

# Thêm các nút cho từng thuật toán
btn_naive_bayes = tk.Button(app, text="Naive Bayes", command=run_naive_bayes, font=("Arial", 12), width=20)
btn_naive_bayes.pack(pady=10)

btn_cart = tk.Button(app, text="CART (Gini Index)", command=run_cart, font=("Arial", 12), width=20)
btn_cart.pack(pady=10)

btn_id3 = tk.Button(app, text="ID3 (Information Gain)", command=run_id3, font=("Arial", 12), width=20)
btn_id3.pack(pady=10)

btn_neural_network = tk.Button(app, text="Neural Network", command=run_neural_network, font=("Arial", 12), width=20)
btn_neural_network.pack(pady=10)

# Chạy ứng dụng
app.mainloop()
