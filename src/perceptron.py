import random as rnd

import numpy as np


class PerceptronSimple:
    def __init__(self, input_size, tita, learning_rate=0.05):
        self.weights = [rnd.uniform(0, 1) for _ in range(input_size + 1)]  # +1 for bias
        self.learning_rate = learning_rate
        self.tita = tita

    def predict(self, x):
        # x must already include the bias term (x0 = 1)
        h = sum(self.weights[i] * x[i] for i in range(len(self.weights)))
        return self.tita(h)

    def set_weights(self, weights):
        if len(weights) != len(self.weights):
            raise ValueError("Los pesos enviados no tienen la longitud correspondiente")
        self.weights = weights.copy()

    def train(self, data, labels, epochs=1000) -> dict:
        timelapse = {"data": data, "labels": labels, "lapse": {}}

        for epoch in range(epochs):
            total_error = 0
            for xi, yi in zip(data, labels):
                y_hat = self.predict(xi)
                delta = yi - y_hat

                for i in range(len(self.weights)):
                    self.weights[i] += self.learning_rate * delta * xi[i]
                total_error += abs(delta)

            timelapse["lapse"][epoch] = {
                "weights": self.weights.copy(),
                "total_error": total_error,
            }

            if total_error == 0:
                break

        return timelapse


class PerceptronLineal:
    def __init__(self, input_size, learning_rate=0.05):
        self.weights = [rnd.uniform(0, 1) for _ in range(input_size + 1)]
        self.learning_rate = learning_rate

    def predict(self, x):
        return sum(self.weights[i] * x[i] for i in range(len(self.weights)))  # sin activación

    def train(self, data, labels, epochs=1000):
        timelapse = {"data": data, "labels": labels, "lapse": {}}

        for epoch in range(epochs):
            total_error = 0
            for xi, yi in zip(data, labels):
                y_hat = self.predict(xi)
                delta = yi - y_hat
                for i in range(len(self.weights)):
                    self.weights[i] += self.learning_rate * delta * xi[i]
                total_error += abs(delta)

            timelapse["lapse"][epoch] = {
                "weights": self.weights.copy(),
                "total_error": total_error,
            }

            if total_error < 1e-3:  # tolerancia para error continuo
                break

        return timelapse


class PerceptronNoLineal:
    def __init__(self, input_size, tita, tita_prime, learning_rate=0.05):
        self.weights = [rnd.uniform(0, 1) for _ in range(input_size + 1)]
        self.learning_rate = learning_rate
        self.tita = tita
        self.tita_prime = tita_prime

    def predict(self, x):
        h = sum(self.weights[i] * x[i] for i in range(len(self.weights)))
        return self.tita(h), self.tita_prime(h)

    def train(self, data, labels, epochs=1000):
        timelapse = {"data": data, "labels": labels, "lapse": {}}

        for epoch in range(epochs):
            total_error = 0
            for xi, yi in zip(data, labels):
                y_hat, y_hat_prime = self.predict(xi)
                delta = yi - y_hat
                for i in range(len(self.weights)):
                    self.weights[i] += self.learning_rate * delta * y_hat_prime * xi[i]
                total_error += abs(delta)

            timelapse["lapse"][epoch] = {
                "weights": self.weights.copy(),
                "total_error": total_error,
            }

            if total_error < 1e-3:
                break

        return timelapse


class PerceptronMulticapa:
    def __init__(self, capas, tita, tita_prime, alpha=0.1):
        self.capas = capas  # lista de tamaños de capas [input, hidden1, ..., output]
        self.alpha = alpha
        self.tita = tita
        self.tita_prime = tita_prime
        self.pesos = []  # pesos[i] conecta capas[i] -> capas[i+1]

        # Inicialización aleatoria de los pesos (incluyendo bias)
        for i in range(len(capas) - 1):
            filas = capas[i + 1]
            columnas = capas[i] + 1  # +1 por bias
            self.pesos.append([[rnd.uniform(-1, 1) for _ in range(columnas)] for _ in range(filas)])

    def forward(self, x):
        entrada = x[:]
        activaciones = [entrada]

        for w in self.pesos:
            entrada = [1] + entrada  # bias
            salida = []
            for neurona in w:
                h = sum(wij * xi for wij, xi in zip(neurona, entrada))
                salida.append(self.tita(h))
            activaciones.append(salida)
            entrada = salida
        return activaciones

    def backward(self, activaciones, y):
        deltas = [None] * len(self.pesos)
        salida = activaciones[-1]
        error = np.subtract(y, salida)

        # Delta de la capa de salida
        deltas[-1] = [e * self.tita_prime(s) for e, s in zip(error, salida)]

        # Delta de las capas ocultas
        for le in reversed(range(len(deltas) - 1)):
            capa = activaciones[le + 1]
            siguiente_delta = deltas[le + 1]
            siguiente_pesos = self.pesos[le + 1]

            deltas[le] = []
            for i in range(len(capa)):
                error_oculto = sum(siguiente_delta[k] * siguiente_pesos[k][i + 1] for k in range(len(siguiente_delta)))
                deltas[le].append(error_oculto * self.tita_prime(capa[i]))

        # Actualización de pesos
        for le in range(len(self.pesos)):
            entrada = [1] + activaciones[le]
            for j in range(len(self.pesos[le])):
                for i in range(len(self.pesos[le][j])):
                    self.pesos[le][j][i] += self.alpha * deltas[le][j] * entrada[i]

    def train(self, datos, salidas, epocas=10000, tolerancia=0.01):
        for epoca in range(epocas):
            error_total = 0
            for x, y in zip(datos, salidas):
                y = y if isinstance(y, list) else [y]
                activaciones = self.forward(x)
                self.backward(activaciones, y)
                error_total += sum((yi - ai) ** 2 for yi, ai in zip(y, activaciones[-1])) / len(y)
            error_promedio = error_total / len(datos)
            if error_promedio < tolerancia:
                print(f"Convergió en la época {epoca} con error {error_promedio}")
                break

    def predict(self, x):
        return self.forward(x)[-1]
