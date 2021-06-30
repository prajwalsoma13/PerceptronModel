import numpy as np

from perceptronModel.perceptron import Perceptron


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

print("Training Perceptron...")
p = Perceptron(X.shape[1])
p.fit(X, y)

print("Testing Perceptron")

for (x, target) in zip(X, y):
    pred = p.predict(x)
    print("data={}, ground-truth={}, pred={}".format(x, target[0], pred))
