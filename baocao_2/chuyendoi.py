import pandas as pd

# Đọc tệp dữ liệu ban đầu
data = pd.read_csv('Bank_Customer_Churn_Prediction.csv')

# Các cột bạn muốn chuyển đổi
columns_to_convert = ['estimated_salary', 'balance', 'age', 'credit_score']

for col in columns_to_convert:
    # Tính min và max của cột
    min_value = data[col].min()
    max_value = data[col].max()

    # Chuyển đổi giá trị theo phạm vi min-max
    data[col] = ((data[col] - min_value) / (max_value - min_value)) * 5 # Chia cho 4 để nằm trong phạm vi 0-4
    data[col] = data[col].round()  # Làm tròn đến số nguyên gần nhất

# Lưu lại tệp dữ liệu đã chuyển đổi
data.to_csv('data_main.csv', index=False)
