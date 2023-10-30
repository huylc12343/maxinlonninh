import tkinter as tk
from tkinter import Label, Entry, Button
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
from tkinter import messagebox  # Thêm thư viện messagebox
from sklearn import preprocessing
import numpy as np

# Đọc dữ liệu và tiền xử lý
df = pd.read_csv('baocao_3\shopping_behavior_updated.csv')
le = preprocessing.LabelEncoder()
df = df.apply(le.fit_transform)
df = np.array(df)
dt_train, dt_Test = train_test_split(df, test_size=0.1, shuffle=True)
X_train = dt_train[:, 1:18]
x_test = dt_Test[:, 1:18]
n_clusters = 5

# Mô hình KMeans
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X_train)

# Tính toán độ đo Silhouette và Davies-Bouldin
silhouette = silhouette_score(X_train, kmeans.labels_)
davies_bouldin = davies_bouldin_score(X_train, kmeans.labels_)

def get_values():
    new_sample = []
    for textbox in textboxes:
        selected_value = textbox.get()
        if not selected_value:  # Kiểm tra nếu ô textbox trống
            messagebox.showerror("Lỗi", "Vui lòng nhập đầy đủ dữ liệu.")
            return
        new_sample.append(selected_value)

    new_sample = list(map(float, new_sample))
    label = kmeans.predict([new_sample])[0]
    result_label.config(text=f'Nhãn: {label}')

# Tạo giao diện
root = tk.Tk()
root.title("Dự đoán phân cụm")
root.geometry("600x700")

form = tk.Frame(root)  # Sử dụng Frame thay vì một cửa sổ riêng lẻ
form.pack()

labels = ["Age: ", "Gender: ", "Item Purchased: ", "Category: ", "Purchase Amount (USD): ", "Location: ",
          "Size: ", "Color: ", "Season: ", "Review Rating: ", "Subscription Status: ", "Shipping Type: ",
          "Discount Applied: ", "Promo Code Used: ", "Previous Purchases: ", "Payment Method: ", "Frequency of Purchases:"]
textboxes = []

for row, label_text in enumerate(labels, start=2):
    label = tk.Label(form, text=label_text)
    label.grid(row=row, column=1, padx=40, pady=5)  # Giảm giá trị pady xuống 5

    textbox = tk.Entry(form)
    textbox.grid(row=row, column=2, pady=5)  # Giảm giá trị pady xuống 5
    textboxes.append(textbox)

# Tạo nút "Lấy giá trị"
get_button = tk.Button(form, text="Dự đoán", command=get_values)
get_button.grid(row=len(labels) + 2, column=1, columnspan=2, pady=10)  # Tăng giá trị pady lên 10

# Hiển thị nhãn dự đoán
result_label = Label(root, text="")
result_label.pack()

# Hiển thị độ đo Silhouette và Davies-Bouldin
Label(root, text=f'Silhouette Score: {silhouette}').pack()
Label(root, text=f'Davies-Bouldin Score: {davies_bouldin}').pack()

root.mainloop()
