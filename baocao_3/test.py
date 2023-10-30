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
df = pd.read_csv('baocao_3\Customers.csv')
le = preprocessing.LabelEncoder()
df = df.apply(le.fit_transform)
df = np.array(df)
dt_train, dt_Test = train_test_split(df, test_size=0.1, shuffle=True)
X_train = dt_train[:, 1:8]
x_test = dt_Test[:, 1:8]
n_clusters = 5

# Mô hình KMeans
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X_train)

# Tính toán độ đo Silhouette và Davies-Bouldin
silhouette = silhouette_score(X_train, kmeans.labels_)
davies_bouldin = davies_bouldin_score(X_train, kmeans.labels_)

print("silhouette ",silhouette)
print("davies_bouldin ",davies_bouldin)