import numpy as np
from abc import ABC, abstractmethod


# Base classifier class
class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        # Abstract method to fit the model with features X and target y
        pass

    @abstractmethod
    def predict(self, X):
        # Abstract method to make predictions on the dataset X
        pass

    @abstractmethod
    def predict_proba(self, X):
        # Abstract method predict the probability of the dataset X
        pass


class DecisionTreeClassifier:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        raise NotImplementedError

    # Split dataset based on a feature and threshold
    def split_dataset(X, y, feature_index, threshold):
        raise NotImplementedError


    # Find the best split for the dataset
    def find_best_split(X, y):
        raise NotImplementedError

    def entropy(y):
        raise NotImplementedError
    

    # prediction
    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def _predict_tree(self, x, tree_node):
        raise NotImplementedError
    

    # print tree
    def print_tree(self, max_print_depth=3):
        raise NotImplementedError