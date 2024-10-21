import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import MLPClassifier, activation, optimizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import os
import sys

def dataPreprocessing():
    """ TODO, use your own dataPreprocess function here. """
    root_path = r"../proj2_data/"

    train_X = os.path.join(root_path, "train_x.csv")
    train_y = os.path.join(root_path, "train_y.csv")
    test_X = os.path.join(root_path, "test_x.csv")
    test_y = os.path.join(root_path, "test_y.csv")

    train_data = pd.read_csv(train_X)
    train_label = pd.read_csv(train_y)
    train_data['y'] = train_label.iloc[:, -1]

    train_X, train_y = Preprocessor(train_data).preprocess()

    test_data = pd.read_csv(test_X)
    test_labels = pd.read_csv(test_y)
    test_data['y'] = test_labels.iloc[:, -1]

    test_X, test_y = Preprocessor(test_data).preprocess()

    return train_X, train_y, test_X, test_y # train, test data should be numpy array


def main():
    train_X, train_y, test_X, test_y = dataPreprocessing() # train, test data should not contain index

    # Define the hyperparameters
    layers = [train_X.shape[1], 64, 32, 16, 1]
    activate_function = activation(type='sigmoid')
    optimizer_choice = optimizer(type='sgd')
    learning_rate = 0.005
    n_epoch = 1000

    # Create the model with the defined hyperparameters
    model = MLPClassifier(layers=layers, activate_function=activate_function, optimizer=optimizer_choice, learning_rate=learning_rate, n_epoch=n_epoch)

    model.fit(train_X, train_y)
    pred = model.predict(test_X)

    acc = accuracy_score(pred, test_y)
    f1 = f1_score(pred, test_y, zero_division=0)
    mcc = matthews_corrcoef(pred, test_y)

    print(f'Acc: {acc:.5f}')
    print(f'F1 score: {f1:.5f}')
    print(f'MCC: {mcc:.5f}')
    scoring = 0.3 * acc + 0.35 * f1 + 0.35 * mcc
    print(f'Scoring: {scoring:.5f}')



if __name__ == "__main__":
    np.random.seed(0)
    main()
    

