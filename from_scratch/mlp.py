"""A feedforward neural network built from scratch: forward pass, manual
backpropagation, and gradient-descent updates, using nothing but numpy.

Architecture: arbitrary stack of Dense(in, out) -> activation layers, trained
with mean-squared-error or binary cross-entropy loss via full-batch or
mini-batch gradient descent.
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(z.dtype)


ACTIVATIONS = {
    "relu": (relu, relu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
}


class DenseLayer:
    def __init__(self, n_in, n_out, activation="relu", rng=None):
        rng = rng or np.random.default_rng()
        # He initialization keeps activation variance stable across layers.
        self.W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        self.b = np.zeros((1, n_out))
        self.activation, self.activation_grad = ACTIVATIONS[activation]

        # Cached for backward()
        self._x = None
        self._z = None

    def forward(self, x):
        self._x = x
        self._z = x @ self.W + self.b
        return self.activation(self._z)

    def backward(self, grad_output, lr):
        # grad_output: dL/d(activation output), shape (batch, n_out)
        grad_z = grad_output * self.activation_grad(self._z)

        grad_W = self._x.T @ grad_z / self._x.shape[0]
        grad_b = grad_z.mean(axis=0, keepdims=True)
        grad_x = grad_z @ self.W.T  # propagate to previous layer

        self.W -= lr * grad_W
        self.b -= lr * grad_b
        return grad_x


class MLP:
    def __init__(self, layer_sizes, activations, seed=0):
        rng = np.random.default_rng(seed)
        assert len(activations) == len(layer_sizes) - 1
        self.layers = [
            DenseLayer(layer_sizes[i], layer_sizes[i + 1], activations[i], rng)
            for i in range(len(layer_sizes) - 1)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_loss, lr):
        grad = grad_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)

    def predict(self, x):
        return self.forward(x)

    def train_step(self, x, y, lr):
        y_pred = self.forward(x)
        loss = binary_cross_entropy(y, y_pred)
        grad_loss = bce_grad(y, y_pred)
        self.backward(grad_loss, lr)
        return loss


def binary_cross_entropy(y_true, y_pred, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def bce_grad(y_true, y_pred, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    # dL/dy_pred for BCE; combined with the final sigmoid's own gradient in
    # DenseLayer.backward, so this is dL/d(network output), not dL/dz.
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])
