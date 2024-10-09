import numpy as np
import pandas as pd

from preprocessor import Preprocessor
from model import LogisticRegressionClassifier
from sklearn.metrics import accuracy_score

def dataPreprocessing():
    train_data = pd.read_csv('train_X.csv')
    train_label = pd.read_csv('train_y.csv')
    train_data['y'] = train_label.iloc[:, -1]

    train_X, train_y = Preprocessor(train_data).preprocess()

    test_data = pd.read_csv('test_X.csv')
    test_labels = pd.read_csv('test_y.csv')
    test_data['y'] = test_labels.iloc[:, -1]

    test_X, test_y = Preprocessor(test_data).preprocess()

    return train_X, train_y, test_X, test_y

def main():
    train_X, train_y, test_X, test_y = dataPreprocessing() # train, test data should not contain index
    model = LogisticRegressionClassifier()
    model.fit(train_X, train_y)

    pred = model.predict(test_X)
    prob = model.predict_proba(test_X)
    prob = [f'{x:.5f}' for x in prob]
    # print(f'Prob: {prob}')
    print(f'Acc: {accuracy_score(pred, test_y):.5f}')


if __name__ == "__main__":
    np.random.seed(0)
    main()
