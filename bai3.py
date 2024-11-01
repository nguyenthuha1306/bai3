import cv2
import numpy as np
import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


# 1. Đọc và tiền xử lý hình ảnh
def read_and_preprocess_image(file_path, target_size=(64, 64)):
    """
    Đọc ảnh từ đường dẫn và chuyển đổi về kích thước target_size.
    Chuyển ảnh sang định dạng xám (grayscale) để đơn giản hóa.
    """
    image = cv2.imread(file_path)
    image = cv2.resize(image, target_size)  # Resize ảnh về kích thước cố định
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển đổi ảnh sang grayscale
    return image.flatten()  # Chuyển ảnh sang dạng vector 1D


# Đọc các hình ảnh đã tải lên
image_1 = read_and_preprocess_image("D:/cho/anh1.jpg")
image_2 = read_and_preprocess_image("D:/cho/anh2.jpg")
image_3 = read_and_preprocess_image("D:/cho/anh3.jpg")
image_4 = read_and_preprocess_image("D:/cho/anh4.jpg")
image_5 = read_and_preprocess_image("D:/cho/image.jpg")

# 2. Tạo tập dữ liệu và nhãn (label)
X = np.array([image_1, image_2, image_3])
y = np.array([0, 1, 2])  # Giả sử: 0 = "Gấu bông", 1 = "Rottweiler", 2 = "Husky"

# Chia tập dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Triển khai các thuật toán và đánh giá
models = {
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(n_neighbors=1),  # Đặt n_neighbors = 1
    "Decision Tree": DecisionTreeClassifier()
}

results = []

for model_name, model in models.items():
    start_time = time.time()  # Ghi nhận thời gian bắt đầu
    model.fit(X_train, y_train)  # Huấn luyện mô hình
    end_time = time.time()  # Ghi nhận thời gian kết thúc

    # Dự đoán và tính toán độ đo
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    elapsed_time = end_time - start_time

    results.append({
        "Model": model_name,
        "Time (s)": elapsed_time,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall
    })

# Chuyển kết quả thành dataframe để hiển thị
results_df = pd.DataFrame(results)
print(results_df)