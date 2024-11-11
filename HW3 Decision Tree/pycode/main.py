import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import DecisionTreeClassifier
import os
def dataPreprocessing():
    root_path = "../proj3_data/"

    train_X_path = os.path.join(root_path, "train_x.csv")
    train_y_path = os.path.join(root_path, "train_y.csv")
    test_X_path = os.path.join(root_path, "test_x.csv")

    train_data = pd.read_csv(train_X_path)
    train_label = pd.read_csv(train_y_path)
    test_data = pd.read_csv(test_X_path)

    train_X = Preprocessor(train_data).preprocess()
    train_y = Preprocessor(train_label).preprocess().flatten().astype(int)
    test_X = Preprocessor(test_data).preprocess()

    return train_X, train_y, test_X


def main():
    train_X, train_y, test_X = dataPreprocessing()
    
    Decision_tree = DecisionTreeClassifier(max_depth=10)
    Decision_tree.fit(train_X, train_y)
    Decision_tree.print_tree(train_y)

    predict_y = Decision_tree.predict(test_X)
    pd.DataFrame(predict_y, columns=['label']).to_csv('../proj3_data/predict_y.csv', index=True)

if __name__ == "__main__":
    np.random.seed(0)
    main()
