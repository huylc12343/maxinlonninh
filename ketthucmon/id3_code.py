import math
import numpy as np
from sklearn import preprocessing
import pandas as pd 
from sklearn.model_selection import train_test_split

def entropy(data):
    # Tính entropy của tập dữ liệu
    total_samples = len(data)
    if total_samples == 0:
        return 0
    
    class_counts = {}
    for row in data:
        label = row[-1]
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    entropy_value = 0
    for count in class_counts.values():
        probability = count / total_samples
        entropy_value -= probability * math.log2(probability)
    
    return entropy_value

def split_data(data, attribute_index):
    # Chia tập dữ liệu thành các phần dựa trên giá trị của thuộc tính
    data_partitions = {}
    for row in data:
        value = row[attribute_index]
        if value not in data_partitions:
            data_partitions[value] = []
        data_partitions[value].append(row)
    return data_partitions

def choose_best_attribute(data, attributes):
    # Chọn thuộc tính tốt nhất để chia
    initial_entropy = entropy(data)
    best_info_gain = 0
    best_attribute = None
    
    for attribute_index in range(len(attributes) - 1):  # -1 để loại trừ thuộc tính nhãn
        data_partitions = split_data(data, attribute_index)
        new_entropy = 0
        for partition in data_partitions.values():
            partition_weight = len(partition) / len(data)
            new_entropy += partition_weight * entropy(partition)
        
        info_gain = initial_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_attribute = attribute_index
    
    return best_attribute

def majority_class(data):
    # Trả về nhãn phổ biến nhất trong tập dữ liệu
    class_counts = {}
    for row in data:
        label = row[-1]
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    return max(class_counts, key=class_counts.get)

def build_tree(data, attributes):
    # Xây dựng cây quyết định
    if not attributes:
        # Không còn thuộc tính để chọn
        return majority_class(data)
    
    if len(set(row[-1] for row in data)) == 1:
        # Tất cả các mẫu thuộc cùng một lớp
        return data[0][-1]
    
    best_attribute_index = choose_best_attribute(data, attributes)
    if best_attribute_index is None:
        # Nếu không còn thuộc tính để chọn, trả về lớp phổ biến nhất
        return majority_class(data)
    
    best_attribute = attributes[best_attribute_index]
    tree = {best_attribute: {}}
    attributes.remove(best_attribute)

    data_partitions = split_data(data, best_attribute_index)
    for value, partition in data_partitions.items():
        subtree = build_tree(partition, attributes[:])
        tree[best_attribute][value] = subtree
    
    return tree

def predict(tree, attributes, sample):
    if isinstance(tree, dict):
        attribute = next(iter(tree))
        subtree = tree[attribute]
        if attribute in attributes:  # Thêm điều kiện kiểm tra thuộc tính có trong danh sách attributes không
            attribute_index = attributes.index(attribute)
            sample_value = sample[attribute_index]
            if sample_value in subtree:
                # Nếu giá trị của thuộc tính trong dữ liệu mẫu nằm trong cây, tiếp tục dự đoán
                return predict(subtree[sample_value], attributes, sample)
            else:
                # Nếu không có giá trị tương ứng trong cây, trả về lớp phổ biến nhất
                return majority_class(data)
        else:
            # Nếu thuộc tính không tồn tại trong danh sách attributes, trả về lớp phổ biến nhất
            return majority_class(data)
    else:
        # Khi đến lá của cây (là một giá trị lớp), trả về giá trị lớp đó
        return tree

# Dữ liệu mẫu
data = pd.read_csv('baocao_2\data_main.csv')

# Đảm bảo thứ tự cột 'churn' đúng
data = data[['credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary', 'churn']]

le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)
data = np.array(data)

# Tách dữ liệu thành tập huấn luyện và tập kiểm tra
dt_train, dt_Test = train_test_split(data, test_size = 0.3, shuffle = True)

# print(df)
X_train = dt_train[:,:10]
y_train = dt_train[:,10]
x_test = dt_Test[:,:10]
y_test = dt_Test[:,10]

# Danh sách tên thuộc tính
attributes = ['credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary']

# Xây dựng cây quyết định
decision_tree = build_tree(np.column_stack((X_train, y_train)), attributes)

# Dự đoán cho các mẫu dữ liệu kiểm tra
y_pred = [predict(decision_tree, attributes, sample) for sample in x_test]

c = 0
for i in range(0, len(y_pred)):
    if(y_test[i] == y_pred[i]):
        c = c + 1
print('ty le du doan dung: ', c/len(y_pred))

print(y_pred)
