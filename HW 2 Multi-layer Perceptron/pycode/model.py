import numpy as np
from abc import ABC, abstractmethod

# ====== Activation funtion ====== #
class activation():
    def __init__(self):
        # TODO
        pass
    

# ====== Optimizer function ====== #
class optimizer():
    def __init__(self):
        # TODO
        pass


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


class MLPClassifier(Classifier):
    def __init__(self, layers, activate_function, optimizer, learning_rate, n_epoch = 1000):
        """ TODO, Initialize your own MLP class """

        self.layers = layers
        self.activate_function = activate_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        
    def forwardPass(self, X):
        """ Forward pass of MLP """
        # TODO
        pass

    def backwardPass(self, y):
        """ Backward pass of MLP """
        # TODO
        pass

    def update(self):
        """ The update method to update parameters """
        # TODO
        pass
    
    def fit(self, X_train, y_train):
        """ Fit method for MLP, call it to train your MLP model """
        # TODO
        pass

    def predict(self, X_test):
        """ Method for predicting class of the testing data """
        y_hat = self.predict_proba(X_test)
        return np.array([1 if i > 0.5 else 0 for i in y_hat])
    
    def predict_proba(self, X_test):
        """ Method for predicting the probability of the testing data """
        return self.forwardPass(X_test)


    