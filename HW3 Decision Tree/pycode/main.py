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

    train_data['y'] = train_label.iloc[:, -1]

    train_X, train_y = Preprocessor(train_data).preprocess(train=True)
    test_X = Preprocessor(test_data).preprocess(train=False)

    return train_X, train_y, test_X

def calculate_metrics(pred, test_y):
    correct_predictions = np.sum(test_y == pred)
    total_predictions = len(test_y)
    acc = correct_predictions / total_predictions

    tp = np.sum((test_y == 1) & (pred == 1))
    tn = np.sum((test_y == 0) & (pred == 0))
    fp = np.sum((test_y == 0) & (pred == 1))
    fn = np.sum((test_y == 1) & (pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0

    score = 0.3 * acc + 0.35 * f1 + 0.35 * mcc
    print(f'score: {score}, acc: {acc}, f1: {f1}, mcc: {mcc}')
    return score

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
        score = calculate_metrics(val_predictions, y_val)

        if score > best_score:
            best_score = score
            best_model = model

    return best_model, best_score


def main():
    train_X, train_y, test_X = dataPreprocessing()
    
    best_model, best_score = k_fold_cross_validation(train_X, train_y, k=5)
    print(f"Best validation score: {best_score}")
    best_model.fit(train_X, train_y)
    best_model.print_tree(train_y)
    predict_y = best_model.predict(test_X)

    pd.DataFrame(predict_y, columns=['label']).to_csv('../proj3_data/predict_y.csv', index=True)

if __name__ == "__main__":
    np.random.seed(0)
    main()
