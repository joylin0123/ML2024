import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import os
def dataPreprocessing():
    root_path = "../proj3_data/"

    train_X_path = os.path.join(root_path, "train_x.csv")
    train_y_path = os.path.join(root_path, "train_y.csv")
    test_X_path = os.path.join(root_path, "test_x.csv")

    train_data = pd.read_csv(train_X_path)
    train_label = pd.read_csv(train_y_path)
    test_data = pd.read_csv(test_X_path)

    train_X = Preprocessor(train_data).preprocess(y=False)
    train_y = Preprocessor(train_label).preprocess().flatten().astype(int)
    test_X = Preprocessor(test_data).preprocess(y=False)

    return train_X, train_y, test_X

def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

def print_metrics(pred, test_y):
    acc = accuracy_score(pred, test_y)
    f1 = f1_score(pred, test_y, zero_division=0)
    mcc = matthews_corrcoef(pred, test_y)
    print(f'Acc: {acc:.5f}')
    print(f'F1 score: {f1:.5f}')
    print(f'MCC: {mcc:.5f}')
    scoring = 0.3 * acc + 0.35 * f1 + 0.35 * mcc
    print(f'Scoring: {scoring:.5f}\n')

def k_fold_cross_validation(X, y, k=5):
    fold_size = len(X) // k
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    best_model = None
    best_score = 0

    for i in range(k):
        val_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        model = DecisionTreeClassifier(max_depth=10)
        model.fit(X_train, y_train)
        model.post_prune(X_val, y_val)
        val_predictions = model.predict(X_val)
        score = calculate_accuracy(y_val, val_predictions)
        print_metrics(val_predictions, y_val)

        if score > best_score:
            best_score = score
            best_model = model

    return best_model, best_score


def main():
    train_X, train_y, test_X = dataPreprocessing()
    
    best_model, best_score = k_fold_cross_validation(train_X, train_y, k=5)
    print(f"Best validation accuracy: {best_score}")
    best_model.fit(train_X, train_y)
    best_model.print_tree(train_y)
    predict_y = best_model.predict(test_X)

    pd.DataFrame(predict_y, columns=['label']).to_csv('../proj3_data/predict_y.csv', index=True)

if __name__ == "__main__":
    np.random.seed(0)
    main()
