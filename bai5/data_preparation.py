import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def load_iris_data():
    # Tải dữ liệu Iris
    iris = load_iris()
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(iris.data)  # Chuẩn hóa dữ liệu
    
    # Phân chia thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, iris.target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def load_dental_images(image_folder='data/images'):
    images, labels = [], []
    filenames = sorted(os.listdir(image_folder))  # Sắp xếp tên file để gán nhãn tuần tự
    total_images = len(filenames)
    midpoint = total_images // 2  # Phân nửa số ảnh

    for idx, filename in enumerate(filenames):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64)).flatten()  # Chuyển đổi kích thước và làm phẳng ảnh
            images.append(img)
            # Gán nhãn: nửa đầu là 0, nửa sau là 1
            label = 0 if idx < midpoint else 1
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    
    # Chuẩn hóa dữ liệu ảnh
    scaler = StandardScaler()
    images = scaler.fit_transform(images)  # Chuẩn hóa ảnh để tăng độ chính xác

    print(f"Đã tải {len(images)} ảnh với nhãn tự động chia làm 2 (0 và 1).")
    
    if len(images) == 0:
        raise ValueError("Không có ảnh nào được tải. Kiểm tra thư mục `data/images`.")
    
    return train_test_split(images, labels, test_size=0.3, random_state=42)
