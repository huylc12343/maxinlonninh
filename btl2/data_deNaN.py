import pandas as pd

# Đọc tập dữ liệu từ tệp CSV
df = pd.read_csv('btl2\water_potability.csv')

# Loại bỏ các hàng chứa dữ liệu NaN
df_cleaned = df.dropna()
print(df_cleaned)
# Lưu dữ liệu đã xử lý vào một tệp CSV mới
df_cleaned.to_csv('btl2\cleaned_data.csv', index=False)
