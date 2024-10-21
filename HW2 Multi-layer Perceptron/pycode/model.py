import numpy as np
from abc import ABC, abstractmethod

# ====== Activation funtion ====== #
class activation:
    def __init__(self, type='relu'):
        self.type = type

    def activate(self, x):
        """ Apply the activation function """
        if self.type == 'relu':
            return np.maximum(0, x)
        elif self.type == 'sigmoid':
            x = np.clip(x, -500, 500)  # Adjust the range based on your specific use case
            return 1 / (1 + np.exp(-x))
        elif self.type == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError('Unsupported activation type.')

    def derivative(self, x):
        """ Derivative of the activation function for backpropagation """
        if self.type == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.type == 'sigmoid':
            sig = self.activate(x)
            return sig * (1 - sig)
        elif self.type == 'tanh':
            return 1 - np.tanh(x)**2
        else:
            raise ValueError('Unsupported activation type.')
    

# ====== Optimizer function ====== #
class optimizer:
    def __init__(self, type='sgd'):
        self.type = type

    def update(self, weights, gradients, learning_rate):
        """ Simple gradient descent update """
        if self.type == 'sgd':
            return weights - learning_rate * gradients
        else:
            raise ValueError('Unsupported optimizer type.')


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
        self.weights = []
        self.bias = []
        self.layer_inputs = []

        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]))
            self.bias.append(np.random.randn(layers[i + 1]))
        self.deltas = [None] * len(self.weights)
        
    def forwardPass(self, X):
        """ Forward pass of MLP """
        self.layer_inputs = [X]
        for i in range(len(self.weights)):
            net = np.dot(self.layer_inputs[-1], self.weights[i]) + self.bias[i]
            output = self.activate_function.activate(net)
            self.layer_inputs.append(output)
        return self.layer_inputs[-1]

    def backwardPass(self, y):
        """ Backward pass of MLP """
        error = self.layer_inputs[-1] - y
        self.deltas[-1] = error * self.activate_function.derivative(self.layer_inputs[-1])

        # Backpropagate the deltas
        for i in range(len(self.deltas) - 2, -1, -1):
            error = np.dot(self.deltas[i + 1], self.weights[i + 1].T)
            self.deltas[i] = error * self.activate_function.derivative(self.layer_inputs[i + 1])

    def update(self):
        """ The update method to update parameters """
        for i in range(len(self.weights)):
            # Gradient of weights with respect to the loss function
            gradient_w = np.dot(self.layer_inputs[i+1].T, self.deltas[i])
            
            # Update the weights using the optimizer
            self.weights[i] = self.optimizer.update(self.weights[i], gradient_w, self.learning_rate)
            
            # Update the bias (mean of deltas across batch dimension)
            self.bias[i] = self.optimizer.update(self.bias[i], np.mean(self.deltas[i], axis=0), self.learning_rate)

    def fit(self, X_train, y_train):
        """ Fit method for MLP, call it to train your MLP model """
        for _ in range(self.n_epoch):
            for X_batch, y_batch in zip(X_train, y_train):
                self.forwardPass(X_batch)
                self.backwardPass(y_batch)
                self.update()

    def predict(self, X_test):
        """ Method for predicting class of the testing data """
        y_hat = self.predict_proba(X_test)
        return np.array([1 if i > 0.5 else 0 for i in y_hat])
    
    def predict_proba(self, X_test):
        """ Method for predicting the probability of the testing data """
        return self.forwardPass(X_test)


    
