"""Simple NumPy-based MLP implementation."""

from __future__ import annotations

import numpy as np


class NumpyMLP:
    """Tiny 1-hidden-layer MLP for scalar regression using numpy."""

    def __init__(self, input_dim: int, hidden: int, output_dim: int):
        rng = np.random.default_rng(0)
        self.W1 = rng.normal(0, 0.1, (input_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros((hidden,), dtype=np.float32)
        self.W2 = rng.normal(0, 0.1, (hidden, output_dim)).astype(np.float32)
        self.b2 = np.zeros((output_dim,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass returning predictions and hidden activations."""
        z1 = x @ self.W1 + self.b1
        a1 = np.tanh(z1)
        y = a1 @ self.W2 + self.b2
        return y, a1

    def step(self, x: np.ndarray, y_true: np.ndarray, lr: float) -> float:
        """Single training step with backpropagation. Returns MSE loss."""
        # forward
        y_pred, a1 = self.forward(x)
        # mse loss
        err = y_pred - y_true
        loss = float(np.mean(err**2))
        # backprop
        dL_dy = (2.0 / x.shape[0]) * err
        dL_dW2 = a1.T @ dL_dy
        dL_db2 = dL_dy.sum(axis=0)
        da1 = dL_dy @ self.W2.T
        dz1 = (1 - a1**2) * da1
        dL_dW1 = x.T @ dz1
        dL_db1 = dz1.sum(axis=0)
        # sgd update
        self.W2 -= lr * dL_dW2
        self.b2 -= lr * dL_db2
        self.W1 -= lr * dL_dW1
        self.b1 -= lr * dL_db1
        return loss
