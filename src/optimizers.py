import numpy as np


class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def initialize(self, weights):
        self.velocity = [np.zeros_like(w, dtype=np.float64) for w in weights]

    def update(self, weights, gradients):
        if self.velocity is None:
            self.initialize(weights)

        for i in range(len(weights)):
            # Ensure arrays are numpy arrays
            weights[i] = np.asarray(weights[i], dtype=np.float64)
            gradients[i] = np.asarray(gradients[i], dtype=np.float64)
            
            # Vectorized operations
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * gradients[i]
            weights[i] += self.velocity[i]


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0  # Time step

    def initialize(self, weights):
        self.m = [np.zeros_like(w, dtype=np.float64) for w in weights]
        self.v = [np.zeros_like(w, dtype=np.float64) for w in weights]

    def update(self, weights, gradients):
        if self.m is None:
            self.initialize(weights)

        self.t += 1
        lr_t = self.learning_rate * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)

        for i in range(len(weights)):
            # Ensure arrays are numpy arrays
            weights[i] = np.asarray(weights[i], dtype=np.float64)
            gradients[i] = np.asarray(gradients[i], dtype=np.float64)
            
            # Vectorized operations
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * np.square(gradients[i])
            
            # Update weights using vectorized operations
            weights[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.epsilon)
