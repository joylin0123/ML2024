import numpy as np
from abc import ABC, abstractmethod

# Base classifier class
class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

class DecisionTreeClassifier(Classifier):
    def __init__(self, max_depth=1, min_samples_split=2, min_samples_leaf=1, min_gain_split=0):
        self.max_depth = max_depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_split = min_gain_split

    def fit(self, X, y):
        if X.shape[0] == 0 or len(y) == 0:
            raise ValueError("Training data cannot be empty.")
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if (depth >= self.max_depth or num_samples < self.min_samples_split or
            num_samples <= self.min_samples_leaf or len(np.unique(y)) == 1):
            leaf_value = self._most_common_label(y)
            return {"leaf": leaf_value, "idxs": np.arange(num_samples)}

        best_split = self.find_best_split(X, y)
        if best_split is None or best_split["gain"] < self.min_gain_split:
            leaf_value = self._most_common_label(y)
            return {"leaf": leaf_value, "idxs": np.arange(num_samples)}

        left_idxs, right_idxs = self.split_dataset(X, y, best_split["feature_index"], best_split["threshold"])
        left_tree = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_tree = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return {
            "feature_index": best_split["feature_index"],
            "threshold": best_split["threshold"],
            "left": left_tree,
            "right": right_tree,
            "left_idxs": left_idxs,
            "right_idxs": right_idxs
        }

    def split_dataset(self, X, y, feature_index, threshold):
        left_idxs = np.where(X[:, feature_index] <= threshold)[0]
        right_idxs = np.where(X[:, feature_index] > threshold)[0]
        return left_idxs, right_idxs

    def find_best_split(self, X, y):
        best_gain = 0
        best_split = None
        num_samples, num_features = X.shape
        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_index, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_split = {"feature_index": feature_index, "threshold": threshold, "gain": gain}
        return best_split

    def entropy(self, y):
        y = np.ravel(y)  # Ensures y is 1-dimensional
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def print_tree(self, y, tree=None, depth=0, max_print_depth=3):
        if tree is None:
            tree = self.tree

        def count_labels(tree):
            if "leaf" in tree:
                label = tree["leaf"]
                return (1, 0) if label == 1 else (0, 1)
            if "left" in tree:
                left_labels = y[tree["left_idxs"]]
                zero_count = sum(label == 0 for label in left_labels)
                one_count = sum(label == 1 for label in left_labels)
            if "right" in tree:
                right_labels = y[tree["right_idxs"]]
                zero_count += sum(label == 0 for label in right_labels)
                one_count += sum(label == 1 for label in right_labels)
            else:
                return 0, 0
            return zero_count, one_count

        if "leaf" in tree or depth >= max_print_depth:
            zero_count, one_count = count_labels(tree)
            print(f"{'  ' * depth}[Leaf] [{zero_count} 0 / {one_count} 1]")
        else:
            zero_count, one_count = count_labels(tree)
            print(f"{'  ' * depth}[F{tree['feature_index']}] "
                f"[{zero_count} 0 / {one_count} 1] <= {tree['threshold']}")

            # Print left and right branches only if depth < max_print_depth
            if depth + 1 < max_print_depth:
                print(f"{'  ' * (depth)}Left:")
                self.print_tree(y, tree["left"], depth + 1, max_print_depth)

                print(f"{'  ' * (depth)}Right:")
                self.print_tree(y, tree["right"], depth + 1, max_print_depth)

    def _information_gain(self, X, y, feature_index, threshold):
        parent_entropy = self.entropy(y)
        left_idxs, right_idxs = self.split_dataset(X, y, feature_index, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        num_samples = len(y)
        num_left, num_right = len(left_idxs), len(right_idxs)
        left_entropy = self.entropy(y[left_idxs])
        right_entropy = self.entropy(y[right_idxs])
        child_entropy = (num_left / num_samples) * left_entropy + (num_right / num_samples) * right_entropy
        return parent_entropy - child_entropy

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def post_prune(self, X_val, y_val):
        def prune_tree(tree, X_val, y_val):
            if "leaf" in tree:
                return tree

            left_tree = prune_tree(tree["left"], X_val, y_val)
            right_tree = prune_tree(tree["right"], X_val, y_val)

            if "leaf" in left_tree and "leaf" in right_tree:
                left_idxs, right_idxs = self.split_dataset(X_val, y_val, tree["feature_index"], tree["threshold"])
                y_val_left = y_val[left_idxs]
                y_val_right = y_val[right_idxs]

                error_before_pruning = np.sum(y_val_left != left_tree["leaf"]) + np.sum(y_val_right != right_tree["leaf"])
                leaf_value = self._most_common_label(np.concatenate([y_val_left, y_val_right]))
                error_after_pruning = np.sum(y_val != leaf_value)

                if error_after_pruning <= error_before_pruning:
                    return {"leaf": leaf_value}

            return {"feature_index": tree["feature_index"], "threshold": tree["threshold"], "left": left_tree, "right": right_tree}
        self.tree = prune_tree(self.tree, X_val, y_val)

    def predict_proba(self, X):
        return np.array([[1, 0] if pred == 1 else [0, 1] for pred in self.predict(X)])

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree_node):
        if "leaf" in tree_node:
            return tree_node["leaf"]
        feature_index = tree_node["feature_index"]
        threshold = tree_node["threshold"]
        if x[feature_index] <= threshold:
            return self._predict_tree(x, tree_node["left"])
        else:
            return self._predict_tree(x, tree_node["right"])
