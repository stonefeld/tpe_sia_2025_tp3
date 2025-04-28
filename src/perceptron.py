import random as rnd


class Perceptron:
    def __init__(self, input_size, tita, learning_rate=0.05):
        self.weights = [rnd.uniform(0, 2) for _ in range(input_size + 1)]  # +1 for bias
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
