import numpy as np
import pandas as pd
from DecisionTree import MyDecisionTree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # Load dữ liệu ví dụ từ một tập dữ liệu
    df = pd.read_csv('ketthucmon\Iris.csv')

    le = preprocessing.LabelEncoder()
    df = df.apply(le.fit_transform)
    df = np.array(df)
    dt_train, dt_Test = train_test_split(df, test_size = 0.3, shuffle = True)

    x_train = dt_train [:,0:3]
    y_train = dt_train [:,4]
    x_test = dt_Test [:,0:3]
    y_test = dt_Test [:,4]
    
    # Khởi tạo và huấn luyện cây quyết định
    decision_tree = MyDecisionTree(criterion="entropy", max_depth=None, min_samples_split=2, min_samples_leaf=1)
    decision_tree.fit(x_train, y_train)
    
    # Dự đoán trên tập kiểm tra
    y_pred = decision_tree.predict(x_test)
    
    # Tính độ chính xác
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Độ chính xác: {accuracy}")

if __name__ == "__main__":
    main()
