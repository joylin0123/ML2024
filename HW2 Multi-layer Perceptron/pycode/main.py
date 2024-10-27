import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import MLPClassifier, activation, Optimizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import os

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

    test_X, test_y = Preprocessor(test_data).preprocess(train=False)

    return train_X, train_y, test_X, test_y # train, test data should be numpy array

def print_metrics(pred, test_y):
    acc = accuracy_score(pred, test_y)
    f1 = f1_score(pred, test_y, zero_division=0)
    mcc = matthews_corrcoef(pred, test_y)
    print(f'Acc: {acc:.5f}')
    print(f'F1 score: {f1:.5f}')
    print(f'MCC: {mcc:.5f}')
    scoring = 0.3 * acc + 0.35 * f1 + 0.35 * mcc
    print(f'Scoring: {scoring:.5f}')

def main():
    train_X, train_y, test_X, test_y = dataPreprocessing() # train, test data should not contain index

    layers = [train_X.shape[1], 128, 64, 32, 1]
    learning_rate = 0.001
    n_epoch = 2000

    activation_types = ['sigmoid', 'tanh', 'relu']
    for act_type in activation_types:
        print(f"\nActivation Function: {act_type}")
        activate_function = activation(type=act_type)
        optimizer_choice = Optimizer(type='sgd')

        model = MLPClassifier(layers=layers,
                              activate_function=activate_function,
                              optimizer=optimizer_choice,
                              learning_rate=learning_rate,
                              n_epoch=n_epoch
                              )
        model.fit(train_X, train_y)
        pred = model.predict(test_X)
        print_metrics(pred, test_y)

    print("\nWithout Activation Function")
    activate_function = activation(type='sigmoid')
    optimizer_choice = Optimizer(type='sgd')
    model = MLPClassifier(layers=layers,
                        activate_function=activate_function,
                        optimizer=optimizer_choice,
                        learning_rate=learning_rate,
                        n_epoch=n_epoch,
                        use_activation=False
                        )
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print_metrics(pred, test_y)

    optimizer_types = ['sgd', 'rmsprop', 'momentum']
    for opt_type in optimizer_types:
        print(f"\nOptimizer: {opt_type}")

        optimizer_choice = Optimizer(type=opt_type)
        model = MLPClassifier(layers=layers,
                              activate_function=activate_function,
                              optimizer=optimizer_choice,
                              learning_rate=learning_rate,
                              n_epoch=n_epoch
                              )
        model.fit(train_X, train_y)
        pred = model.predict(test_X)
        print_metrics(pred, test_y)

    for layer in [[train_X.shape[1], 100, 100, 1], [train_X.shape[1], 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1]]:
        print(f"\nLayers: {layer}")
        activate_function = activation(type='sigmoid')
        optimizer_choice = Optimizer(type='sgd')

        model = MLPClassifier(layers=layer,
                              activate_function=activate_function,
                              optimizer=optimizer_choice,
                              learning_rate=learning_rate,
                              n_epoch=1000
                              )
        model.fit(train_X, train_y)
        pred = model.predict(test_X)
        print_metrics(pred, test_y)

if __name__ == "__main__":
    np.random.seed(0)
    main()
