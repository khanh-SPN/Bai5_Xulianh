# Ứng dụng Phân Lớp Ảnh

Ứng dụng này cho phép người dùng lựa chọn các thuật toán phân lớp khác nhau (Naive Bayes, CART, ID3, và Mạng Neural) để phân loại dữ liệu ảnh và dữ liệu mẫu (Iris). Đây là sản phẩm của **Ngô Duy Khánh**.

## Tổng quan
Ứng dụng cung cấp giao diện sử dụng `tkinter` cho phép người dùng lựa chọn một thuật toán và xem kết quả phân loại, bao gồm độ chính xác và báo cáo phân lớp chi tiết.

### Các tính năng
- **Chọn thuật toán phân lớp**: Người dùng có thể lựa chọn giữa các thuật toán Naive Bayes, CART (Gini Index), ID3 (Information Gain), và Mạng Neural.
- **Hiển thị kết quả chi tiết**: Báo cáo phân lớp và độ chính xác được hiển thị trên giao diện.
- **Tự động gán nhãn** cho ảnh trong thư mục `data/images`.
- **Độ chính xác cao** với dữ liệu mẫu (Iris) và dữ liệu ảnh.

## Ảnh minh họa
Dưới đây là một số ảnh minh họa về giao diện và kết quả phân loại:

- ![Demo 1](./demo/demo1.jpg)
- ![Demo 2](./demo/demo2.jpg)
- ![Demo 3](./demo/demo3.jpg)
- ![Demo 4](./demo/demo4.jpg)

## Yêu cầu hệ thống
- Python 3.6 trở lên
- Các thư viện: `tkinter`, `sklearn`, `tensorflow`, `opencv-python`, `numpy`, `scikit-learn`, `pillow`

## Hướng dẫn cài đặt và chạy chương trình
1. **Clone dự án**:
    ```bash
    git clone https://github.com/yourusername/phanlopanh.git
    cd phanlopanh
    ```

2. **Cài đặt các thư viện cần thiết**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Chạy ứng dụng**:
    ```bash
    python app.py
    ```

4. **Hướng dẫn sử dụng**:
   - Chọn thuật toán mong muốn bằng cách nhấn vào nút tương ứng.
   - Quan sát kết quả báo cáo phân lớp và độ chính xác trên giao diện.

## Cải thiện độ chính xác
Để đảm bảo độ chính xác cao nhất cho các mô hình, các bước sau đã được áp dụng:

1. **Chuẩn hóa dữ liệu**: Áp dụng chuẩn hóa dữ liệu để tăng hiệu suất mô hình.
2. **Điều chỉnh siêu tham số**: Tối ưu hóa tham số của mô hình CART và Mạng Neural để đạt độ chính xác tối đa.

### Ví dụ điều chỉnh siêu tham số
- **Naive Bayes**: Đảm bảo phân phối chuẩn của dữ liệu để đạt kết quả tốt nhất.
- **Mạng Neural**: Sử dụng `epochs` và `batch_size` phù hợp, ví dụ:
    ```python
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))
    ```

## Tác giả
Chương trình này được phát triển bởi **Ngô Duy Khánh**.

## Liên hệ
Nếu có bất kỳ câu hỏi hoặc vấn đề nào, vui lòng liên hệ qua email: `khanhaovl18@example.com`.
