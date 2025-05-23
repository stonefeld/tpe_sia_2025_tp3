from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def initialize(self, weights):
        pass

    @abstractmethod
    def update(self, weights, weight_gradients):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def initialize(self, _):
        pass

    def update(self, _, weights, weight_gradients):
        weights += self.learning_rate * weight_gradients


class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def initialize(self, weights):
        self.velocity = [np.zeros_like(w, dtype=np.float64) for w in weights]

    def update(self, i, weights, weight_gradients):
        self.velocity[i] = self.momentum * self.velocity[i] + self.learning_rate * weight_gradients
        weights += self.velocity[i]


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, layers=None):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        self.simple = False

        if layers is not None and len(layers) <= 3 and max(layers) <= 10:
            self.beta1 = 0.8
            self.beta2 = 0.9
            self.epsilon = 1e-6
            self.learning_rate *= 0.1
            self.simple = True

    def initialize(self, weights):
        self.m = [np.zeros_like(w, dtype=np.float64) for w in weights]
        self.v = [np.zeros_like(w, dtype=np.float64) for w in weights]

    def update(self, i, weights, weight_gradients):
        self.t += 1

        # Actualizar los momentos
        self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * weight_gradients
        self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * weight_gradients**2

        # Corrección de sesgo
        m_hat = self.m[i] / (1 - self.beta1**self.t)
        v_hat = self.v[i] / (1 - self.beta2**self.t)

        # Actualizar los pesos
        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # Limitar el tamaño de los pesos para problemas simples
        if self.simple:
            update = np.clip(update, -0.1, 0.1)

        weights += update
