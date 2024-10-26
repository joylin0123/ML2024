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
        if self.type == 'sigmoid':
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
        if self.type == 'tanh':
            return np.tanh(x)
        raise ValueError('Unsupported activation type.')

    def derivative(self, x):
        """ Derivative of the activation function for backpropagation """
        if self.type == 'relu':
            return np.where(x > 0, 1, 0)
        if self.type == 'sigmoid':
            sig = self.activate(x)
            return sig * (1 - sig)
        if self.type == 'tanh':
            return 1 - np.tanh(x)**2
        raise ValueError('Unsupported activation type.')
    

# ====== Optimizer function ====== #
class Optimizer:
    def __init__(self, type='sgd'):
        self.type = type
        self.momentum = 0.9
        self.epsilon = 1e-8
        self.cache = {}
        self.velocity = {}

    def update(self, weights, gradients, learning_rate):
        """ Update weights based on the optimizer type """
        if self.type == 'sgd':
            return weights - learning_rate * gradients
        if self.type == 'momentum':
            if 'velocity' not in self.velocity:
                self.velocity['weights'] = np.zeros_like(weights)
            self.velocity['weights'] = self.momentum * self.velocity['weights'] + gradients
            return weights - learning_rate * self.velocity['weights']
        if self.type == 'rmsprop':
            if 'cache' not in self.cache:
                self.cache['weights'] = np.zeros_like(weights)
            self.cache['weights'] = 0.9 * self.cache['weights'] + 0.1 * (gradients ** 2)
            return weights - learning_rate * gradients / (np.sqrt(self.cache['weights']) + self.epsilon)
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


# class MLPClassifier(Classifier):
#     def __init__(self, layers, activate_function, optimizer, learning_rate, n_epoch = 1000):
#         """ TODO, Initialize your own MLP class """
#         self.layers = layers
#         self.activate_function = activate_function
#         self.optimizer = optimizer
#         self.learning_rate = learning_rate
#         self.n_epoch = n_epoch
#         self.weights = []
#         self.bias = []
#         self.layer_inputs = []

#         for i in range(len(layers) - 1):
#             self.weights.append(np.random.randn(layers[i], layers[i + 1]))
#             self.bias.append(np.random.randn(layers[i + 1]))

#         self.deltas = [None] * len(self.weights)
        
#     def forwardPass(self, X):
#         """ Forward pass of MLP """
#         self.layer_inputs = [X]

#         for i in range(len(self.weights)):
#             net = np.dot(self.layer_inputs[-1], self.weights[i]) + self.bias[i]
#             output = self.activate_function.activate(net)
#             self.layer_inputs.append(output)
#             self.layer_inputs.append(net)

#         return self.layer_inputs[-1]

#     def backwardPass(self, y):
#         """ Backward pass of MLP """
#         error = self.layer_inputs[-1] - y
#         self.deltas[-1] = error * self.activate_function.derivative(self.layer_inputs[-1])

#         for i in range(len(self.deltas) - 2, -1, -1):
#             error = np.dot(self.deltas[i + 1], self.weights[i + 1].T)
#             self.deltas[i] = error * self.activate_function.derivative(self.layer_inputs[i + 1])

#     def update(self):
#         """ The update method to update parameters """
#         for i in range(len(self.weights)):
#             gradient_w = np.dot(self.layer_inputs[i+1].T, self.deltas[i])
#             self.weights[i] = self.optimizer.update(self.weights[i], gradient_w, self.learning_rate)
#             self.bias[i] = self.optimizer.update(self.bias[i], np.mean(self.deltas[i], axis=0), self.learning_rate)

#     def fit(self, X_train, y_train):
#         """ Fit method for MLP, call it to train your MLP model """
#         for _ in range(self.n_epoch):
#             for X_batch, y_batch in zip(X_train, y_train):
#                 self.forwardPass(X_batch)
#                 self.backwardPass(y_batch)
#                 self.update()

#     def predict(self, X_test):
#         """ Method for predicting class of the testing data """
#         y_hat = self.predict_proba(X_test)
#         return np.array([1 if i > 0.5 else 0 for i in y_hat])
    
#     def predict_proba(self, X_test):
#         """ Method for predicting the probability of the testing data """
#         return self.forwardPass(X_test)

class MLPClassifier(Classifier):
    def __init__(self, layers, activate_function, optimizer, learning_rate, n_epoch=1000, use_activation=True):
        """ Initialize the MLP class with layers and optional activation usage """
        self.layers = layers
        self.activate_function = activate_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.use_activation = use_activation
        self.weights = []
        self.bias = []
        self.layer_inputs = []

        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]))
            self.bias.append(np.random.randn(layers[i + 1]))

        self.deltas = [None] * len(self.weights)
        
    def forwardPass(self, X):
        """ Forward pass of MLP, with optional activation """
        self.layer_inputs = [X]

        for i in range(len(self.weights)):
            net = np.dot(self.layer_inputs[-1], self.weights[i]) + self.bias[i]
            
            if self.use_activation:
                output = self.activate_function.activate(net)
            else:
                output = net

            self.layer_inputs.append(output)

        return self.layer_inputs[-1]

    def backwardPass(self, y):
        """ Backward pass of MLP with optional activation derivative """
        error = self.layer_inputs[-1] - y
        if self.use_activation:
            self.deltas[-1] = error * self.activate_function.derivative(self.layer_inputs[-1])
        else:
            self.deltas[-1] = error

        for i in range(len(self.deltas) - 2, -1, -1):
            error = np.dot(self.deltas[i + 1], self.weights[i + 1].T)
            if self.use_activation:
                self.deltas[i] = error * self.activate_function.derivative(self.layer_inputs[i + 1])
            else:
                self.deltas[i] = error

    def update(self):
        """ Update weights and biases using optimizer """
        for i in range(len(self.weights)):
            gradient_w = np.dot(self.layer_inputs[i + 1].T, self.deltas[i])
            self.weights[i] = self.optimizer.update(self.weights[i], gradient_w, self.learning_rate)
            self.bias[i] = self.optimizer.update(self.bias[i], np.mean(self.deltas[i], axis=0), self.learning_rate)

    def fit(self, X_train, y_train):
        """ Train the MLP model """
        for _ in range(self.n_epoch):
            for X_batch, y_batch in zip(X_train, y_train):
                self.forwardPass(X_batch)
                self.backwardPass(y_batch)
                self.update()

    def predict(self, X_test):
        """ Predict class labels for test data """
        y_hat = self.predict_proba(X_test)
        return np.array([1 if i > 0.5 else 0 for i in y_hat])
    
    def predict_proba(self, X_test):
        """ Predict probability scores for test data """
        return self.forwardPass(X_test)
