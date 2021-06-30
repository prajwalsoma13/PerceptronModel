import json
import numpy as np

from perceptronModel.Configuration import *


class Perceptron:
    def __init__(self, N, alpha=learningRate):
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    @staticmethod
    def step(x):
        """
        :param x: Input's a whole number
        :return: return's 1 or 0 depending on the input value
        """
        return 1 if x > 0 else 0

    def fit(self, X, y, e=epochs):
        """
        :param X: Input's to the perceptron.
        :param y: Predicted values.
        :param e: No of times the input is passed through the perceptron.
        :return: updates the weight.
        """

        """Inserts a column of 1's as the last entry in the feature
           matrix -- this allows us to treat bais as a trainable parameter with the matrix"""
        X = np.c_[X, np.ones((X.shape[0]))]

        for _ in np.arange(0, e):
            for (x, target) in zip(X, y):

                p = self.step(np.dot(x, self.W))

                """updates weight"""
                if p != target:
                    """determine errors"""
                    error = p - target

                    self.W += -self.alpha * error * x

    def predict(self, X, addBias=True):
        """
        :param X: Input matrix.
        :param addBias: adding a threshold to fulfill the criteria.
        :return: Predicts the output.
        """
        X = np.atleast_2d(X)

        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]

        return self.step(np.dot(X, self.W))
