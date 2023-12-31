import math
import numpy as np
from sklearn import preprocessing
import pandas as pd 
from sklearn.model_selection import train_test_split
# import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import precision_score
from sklearn import preprocessing

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
data = pd.read_csv('ketthucmon\Iris.csv')

# Đảm bảo thứ tự cột 'churn' đúng
data = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']]
attributes = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)
data = np.array(data)
# Xây dựng cây quyết định
X = data[:, 0:3]  # Chọn cột 0, 1, và 2 làm features
y = data[:, 4]    # Chọn cột 4 làm target

decision_tree = build_tree(np.column_stack((X, y)), attributes)
precisions = []
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=None)

for train_index, validation_index in kf.split(X):
    X_train, x_test = X[train_index], X[validation_index]
    y_train, y_test = y[train_index], y[validation_index]


    y_pred = [predict(decision_tree, attributes, sample) for sample in x_test]
    precision = precision_score(y_test, y_pred, average='micro')
    precisions.append(precision)

avr_accuracy = sum(precisions) / k
print("Độ chính xác trung bình:", avr_accuracy)