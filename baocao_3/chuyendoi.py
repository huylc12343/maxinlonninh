import pandas as pd
from sklearn import preprocessing

# Đọc tệp dữ liệu ban đầu
data = pd.read_csv('shopping_behavior_updated.csv')

# Các cột bạn muốn chuyển đổi
columns_to_convert = ["Age", "Gender", "Item Purchased", "Category", "Location", "Size", "Color", "Season", "Subscription Status", "Shipping Type", "Discount Applied", "Promo Code Used", "Payment Method", "Frequency of Purchases"]

# Lặp qua các cột và chuyển đổi giá trị
for column in columns_to_convert:
    # Kiểm tra nếu cột chứa kiểu dữ liệu string
    if data[column].dtype == 'object':
        # Sử dụng LabelEncoder để chuyển đổi giá trị thành số nguyên
        le = preprocessing.LabelEncoder()
        data[column] = le.fit_transform(data[column]) + 1

# Lưu lại tệp dữ liệu đã chuyển đổi
data.to_csv('data_main.csv', index=False)
