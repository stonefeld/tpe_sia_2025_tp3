import numpy as np

from src.optimizers import SGD


class PerceptronSimple:
    def __init__(self, input_size, tita, learning_rate=0.05):
        self.weights = np.random.uniform(-1, 1, input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.tita = tita

    def predict(self, x):
        h = np.dot(self.weights, x)
        return self.tita(h)

    def train(self, data, labels, epochs=1000) -> dict:
        data = np.array(data)
        labels = np.array(labels)
        timelapse = {"data": data.tolist(), "labels": labels.tolist(), "lapse": {}}

        for epoch in range(epochs):
            total_error = 0
            for xi, yi in zip(data, labels):
                y_hat = self.predict(xi)
                delta = yi - y_hat
                self.weights += self.learning_rate * delta * xi
                total_error += abs(delta)

            timelapse["lapse"][epoch] = {
                "weights": self.weights.tolist(),
                "total_error": float(total_error),
            }

            if total_error == 0:
                break

        return timelapse


class PerceptronLineal:
    def __init__(self, input_size, learning_rate=0.05):
        self.weights = np.random.uniform(-0.1, 0.1, input_size + 1)
        self.learning_rate = learning_rate

    def predict(self, x):
        data = np.concatenate(([1], x))
        return np.dot(self.weights, data)  # sin activación

    def train(self, data, labels, epochs=1000):
        timelapse = {"data": data.tolist(), "labels": labels.tolist(), "lapse": {}}

        for epoch in range(epochs):
            total_error = 0
            for xi, yi in zip(data, labels):
                y_hat = self.predict(xi)
                delta = yi - y_hat
                xb = np.concatenate(([1], xi))
                self.weights += self.learning_rate * delta * xb
                total_error += 0.5 * (delta**2)  # Mean squared error

            timelapse["lapse"][epoch] = {
                "weights": self.weights.tolist(),
                "total_error": float(total_error),
            }

            total_error /= len(data)

            if total_error < 1e-3:
                break

        return timelapse


class PerceptronNoLineal:
    def __init__(self, input_size, tita, tita_prime, learning_rate=0.05):
        self.weights = np.random.uniform(-0.1, 0.1, input_size + 1)
        self.learning_rate = learning_rate
        self.tita = tita
        self.tita_prime = tita_prime

    def predict(self, x):
        data = np.concatenate(([1], x))
        h = np.dot(self.weights, data)
        return self.tita(h), self.tita_prime(h)

    def train(self, data, labels, epochs=1000, tolerance=1e-3):
        timelapse = {"data": data.tolist(), "labels": labels.tolist(), "lapse": {}}

        for epoch in range(epochs):
            total_error = 0
            for xi, yi in zip(data, labels):
                y_hat, y_hat_prime = self.predict(xi)
                delta = yi - y_hat
                xb = np.concatenate(([1], xi))
                self.weights += self.learning_rate * delta * y_hat_prime * xb
                total_error += 0.5 * (delta**2)  # Mean squared error

            timelapse["lapse"][epoch] = {
                "weights": self.weights.tolist(),
                "total_error": float(total_error),
            }

            total_error /= len(data)

            if total_error < tolerance:
                break

        return timelapse


class PerceptronMulticapa:
    def __init__(self, capas, tita, tita_prime, optimizer=SGD()):
        self.capas = capas  # lista de tamaños de capas [input, hidden1, ..., output]
        self.tita = tita
        self.tita_prime = tita_prime
        self.weights = []  # pesos[i] conecta capas[i] -> capas[i+1]
        self.optimizer = optimizer

        # Inicialización Xavier/Glorot de los pesos
        for i in range(len(capas) - 1):
            neuronas = capas[i + 1]
            entradas = capas[i] + 1  # +1 por bias
            # Xavier/Glorot initialization: scale = sqrt(2.0 / (fan_in + fan_out))
            scale = np.sqrt(2.0 / (entradas + neuronas))
            self.weights.append(np.random.normal(0, scale, (neuronas, entradas)))

        # Inicialización del optimizador
        self.optimizer.initialize(self.weights)

    def forward(self, x):
        entrada = np.array(x)
        activaciones = [entrada]

        for capa in self.weights:
            entrada = np.concatenate(([1], entrada))  # bias
            h = np.dot(capa, entrada)
            salida = np.array([self.tita(h_i) for h_i in h])
            activaciones.append(salida)
            entrada = salida

        return activaciones

    def backward(self, activaciones, y):
        deltas = [None] * len(self.weights)
        salida = activaciones[-1]
        error = np.array(y) - salida

        # Delta de la capa de salida
        deltas[-1] = error * np.array([self.tita_prime(s) for s in salida])

        # Delta de las capas ocultas
        for i in reversed(range(len(deltas) - 1)):
            j = i + 1
            capa = activaciones[j]
            siguiente_delta = deltas[j]
            siguiente_pesos = self.weights[j]

            # Calcular el delta sin considerar el bias
            error_oculto = np.dot(siguiente_delta, siguiente_pesos[:, 1:])
            deltas[i] = error_oculto * np.array([self.tita_prime(s) for s in capa])

        # Cálculo de gradientes
        for i in range(len(self.weights)):
            entrada = np.append(1, activaciones[i])
            weight_gradients = np.outer(deltas[i], entrada)
            self.optimizer.update(i, self.weights[i], weight_gradients)

    def train(self, datos, salidas, epocas=1000, tolerancia=0.01):
        datos = np.array(datos)
        salidas = np.array(salidas)

        for epoca in range(epocas):
            print(f"Época {epoca + 1}/{epocas}...", end=" ")

            error_total = 0
            for x, y in zip(datos, salidas):
                activaciones = self.forward(x)
                self.backward(activaciones, y)
                error_total += np.sum((y - activaciones[-1]) ** 2) / (2 * y.size)

            error_promedio = error_total / len(datos)
            if error_promedio < tolerancia:
                print(f"Convergió en la época {epoca + 1} con error {error_promedio}")
                break

            print(f"Error promedio: {error_promedio}")

    def predict(self, x: list[float]) -> list[float]:
        return self.forward(x)[-1]
