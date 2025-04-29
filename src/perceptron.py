import random as rnd


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
