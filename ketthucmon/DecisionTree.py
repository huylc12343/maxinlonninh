import numpy as np

class MyDecisionTree:
    def __init__(self, criterion="entropy", max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if criterion not in ['entropy', 'gini']:
            raise ValueError('Invalid value for parameter criterion!')
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(set(y)) == 1 or len(X) < self.min_samples_split:
            return max(y, key=list(y).count)  # Trả về nhãn phổ biến nhất
        if all(label == y[0] for label in y):
            return y[0]
        best_attribute = self._choose_best_attribute(X, y)
        if best_attribute is None:
            return max(y, key=list(y).count)
        tree = {best_attribute: {}}
        unique_values = np.unique(X[:, best_attribute])
        for value in unique_values:
            subset_indices = X[:, best_attribute] == value
            X_subset = X[subset_indices]
            y_subset = y[subset_indices]
            tree[best_attribute][value] = self._build_tree(X_subset, y_subset, depth + 1)
        return tree

    def _choose_best_attribute(self, X, y):
        best_attribute = None
        if self.criterion == 'entropy':
            best_information_gain = -1
            for attribute in range(X.shape[1]):
                information_gain = self._calculate_information_gain(X, y, attribute)
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_attribute = attribute
        elif self.criterion == 'gini':
            best_gini_index = float('inf')
            for attribute in range(X.shape[1]):
                information_gain = self._calculate_information_gain(X, y, attribute)
                if information_gain < best_gini_index:
                    best_gini_index = information_gain
                    best_attribute = attribute
        return best_attribute


    def _calculate_information_gain(self, X, y, attribute):
        if self.criterion == 'entropy':
            total_entropy = self._entropy(y)  # Entropy của tập dữ liệu gốc
            unique_values = np.unique(X[:, attribute])
            new_entropy = 0
            for value in unique_values:
                subset_indices = X[:, attribute] == value
                subset_y = y[subset_indices]
                weight = len(subset_y) / len(y)
                new_entropy += weight * self._entropy(subset_y)
            information_gain = total_entropy - new_entropy
            return information_gain
        elif self.criterion == 'gini':
            total_gini = self._gini(y)  # Gini Impurity của tập dữ liệu gốc
            unique_values = np.unique(X[:, attribute])
            new_gini = 0
            for value in unique_values:
                subset_indices = X[:, attribute] == value
                subset_y = y[subset_indices]
                weight = len(subset_y) / len(y)
                new_gini += weight * self._gini(subset_y)
            information_gain = total_gini - new_gini
            return information_gain

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        class_counts = np.bincount(y)
        class_probabilities = class_counts / len(y)
        entropy = -np.sum(class_probabilities * np.log2(class_probabilities + np.finfo(float).eps))
        return entropy

    def _gini(self, y):
        if len(y) == 0:
            return 0
        class_counts = np.bincount(y)
        class_probabilities = class_counts / len(y)
        gini = 1 - np.sum(class_probabilities ** 2)
        return gini

    def predict(self, X):
        predictions = []
        for sample in X:
            prediction = self._traverse_tree(self.tree, sample)
            predictions.append(prediction)
        return predictions

    def _traverse_tree(self, node, sample):
        if isinstance(node, dict):
            attribute = list(node.keys())[0]
            value = sample[attribute]

            if value in node[attribute]:
                # Nếu giá trị thuộc tính nằm trong cây, tiếp tục đệ quy
                return self._traverse_tree(node[attribute][value], sample)
            else:
                # Nếu giá trị thuộc tính không tồn tại trong cây, trả về giá trị mặc định (hoặc phổ biến nhất)
                return self._get_mosat_common_value(node[attribute])
        else:
            # Khi đạt đến lá của cây (là một giá trị nhãn), trả về giá trị đó
            return node
    def _get_most_common_value(self, subtree):
            # Trả về giá trị phổ biến nhất trong một nhánh
            counts = np.bincount(list(subtree.values()))
            return np.argmax(counts)
