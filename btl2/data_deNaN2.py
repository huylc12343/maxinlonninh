import pandas as pd

# Đọc tập dữ liệu từ tệp CSV
df = pd.read_csv('btl2\water_potability.csv')

# Loại bỏ các hàng chứa dữ liệu NaN
df_filled = df.fillna(df.mean())
print(df_filled)
# Lưu dữ liệu đã xử lý vào một tệp CSV mới
df_filled.to_csv('btl2\cleaned_data2.csv', index=False)
