import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder

def entropy(data):
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
    data_partitions = {}
    for row in data:
        value = row[attribute_index]
        if value not in data_partitions:
            data_partitions[value] = []
        data_partitions[value].append(row)
    return data_partitions

def choose_best_attribute(data, attributes):
    initial_entropy = entropy(data)
    best_info_gain = 0
    best_attribute = None
    
    for attribute_index in range(len(attributes) - 1):
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
    class_counts = {}
    for row in data:
        label = row[-1]
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    return max(class_counts, key=class_counts.get)

def build_tree(data, attributes):
    if not attributes.any():
        return majority_class(data)
    
    if len(set(row[-1] for row in data)) == 1:
        return data[0][-1]
    
    best_attribute_index = choose_best_attribute(data, attributes)
    if best_attribute_index is None:
        return majority_class(data)
    
    best_attribute = attributes[best_attribute_index]
    tree = {best_attribute: {}}
    attributes = attributes.drop(best_attribute)

    data_partitions = split_data(data, best_attribute_index)
    for value, partition in data_partitions.items():
        subtree = build_tree(partition, attributes)
        tree[best_attribute][value] = subtree
    
    return tree

def predict(tree, sample):
    if isinstance(tree, dict):
        attribute = next(iter(tree))
        subtree = tree[attribute]
        attribute_value = sample[attribute]
        if attribute_value in subtree:
            return predict(subtree[attribute_value], sample)
        else:
            return majority_class(list(subtree.values())[0])
    else:
        return tree

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv('ketthucmon\Iris.csv')
data = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']]

# Chuyển cột 'Species' thành dữ liệu số
le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'])

# Chia dữ liệu thành features (X) và target (y)
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['Species']

# K-Fold Cross-Validation
precisions = []
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=None)

for train_index, validation_index in kf.split(X):
    X_train, x_test = X.iloc[train_index], X.iloc[validation_index]
    y_train, y_test = y.iloc[train_index], y.iloc[validation_index]

    # Xây dựng cây quyết định và dự đoán trên tập kiểm tra
    decision_tree = build_tree(pd.concat([X_train, y_train], axis=1).values, X_train.columns)
    y_pred = [predict(decision_tree, sample) for _, sample in x_test.iterrows()]
    
    # Tính độ chính xác
    precision = precision_score(y_test, y_pred, average='micro')
    precisions.append(precision)

# Tính độ chính xác trung bình
avr_accuracy = sum(precisions) / k
print("Độ chính xác trung bình:", avr_accuracy)