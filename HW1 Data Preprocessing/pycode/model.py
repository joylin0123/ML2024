import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold

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


# Logistic Regression class from scratch
# !!! You should only modify the parameters of the __init__ function !!!
class LogisticRegressionClassifier(Classifier):
    def __init__(self, C=1.0, penalty='l2', lr=1e-2, iterations=600):
        super(LogisticRegressionClassifier, self).__init__()

        self.iterations = iterations
        self.lr = lr
        self.C = C
        self.penalty = penalty

    def sigmoid(self, x):
        """ The sigmoid function """
        return 1.0 / (1.0 + np.exp(-x))

    def linear(self, X):
        return np.dot(X, self.W) + self.b
    
    def SGD(self, dW, db):
        self.W = self.W - self.lr * dW
        self.b = self.b - self.lr * db

    def binaryCrossEntropy(self, pred, target):
        return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))

    def fit(self, X, y):
        m, n = X.shape # (m, n)
        self.W = np.random.randn(n) # (n,)
        # self.W = np.zeros(n)
        self.b = 0
        self.loss = []

        y = np.array(y).flatten()  # make sure y is one dimension

        for i in range(self.iterations):
            z = self.linear(X)
            y_hat = self.sigmoid(z)

            dW = 1/m * np.dot(X.T, (y_hat - y))
            db = 1/m * np.sum(y_hat - y)

            if self.penalty == 'l2':
                dW += (self.C / m) * self.W
            elif self.penalty == 'l1':
                dW += (self.C / m) * np.sign(self.W)

            # update weight and bias
            self.SGD(dW, db)

            # compute loss
            loss = self.binaryCrossEntropy(y_hat, y)
            self.loss.append(loss)
    
    def predict(self, X):
        y_hat = self.predict_proba(X)
        return [1 if i > 0.5 else 0 for i in y_hat]
    
    def predict_proba(self, X):
        z = self.linear(X)
        return self.sigmoid(z)

    
    def score(self, X, y):
        pred = self.predict(X)
        return accuracy_score(pred, y)
        


# if __name__ == "__main__":
#     iris = datasets.load_iris()
#     X = iris.data
#     y = iris.target

#     train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)
#     model = LogisticRegressionClassifier(iterations=5000, penalty=None)
#     model.fit(train_X, train_y)
#     pred = np.array(model.predict(test_X))
#     acc = accuracy_score(pred, test_y)
#     print(acc)
    # print(f'Weight: {model.W}')
    # print(f'bias: {model.b}')
    