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


class PerceptronMulticapa:
    def __init__(self, input_size, hidden_size, tita, tita_prime, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.tita = tita
        self.tita_prime = tita_prime

        # Pesos capa entrada -> oculta (bias incluido)
        self.w_input_hidden = [[rnd.uniform(-1, 1) for _ in range(input_size + 1)] for _ in range(hidden_size)]

        # Pesos capa oculta -> salida (bias incluido)
        self.w_hidden_output = [rnd.uniform(-1, 1) for _ in range(hidden_size + 1)]

    def _forward(self, x):
        x = [1] + x  # bias input

        # Capa oculta
        hidden_net = [sum(w * xi for w, xi in zip(weights, x)) for weights in self.w_input_hidden]
        hidden_out = [self.tita(h) for h in hidden_net]
        hidden_out_with_bias = [1] + hidden_out  # bias hidden

        # Capa salida
        output_net = sum(w * ho for w, ho in zip(self.w_hidden_output, hidden_out_with_bias))
        output_out = self.tita(output_net)

        return {
            "input": x,
            "hidden_net": hidden_net,
            "hidden_out": hidden_out,
            "hidden_out_with_bias": hidden_out_with_bias,
            "output_net": output_net,
            "output_out": output_out,
        }

    def predict(self, x):
        return self._forward(x)["output_out"]

    def train(self, data, labels, epochs=10000):
        history = []

        for epoch in range(epochs):
            total_error = 0

            for x, y_true in zip(data, labels):
                fwd = self._forward(x)
                y_pred = fwd["output_out"]
                error = y_true - y_pred
                total_error += error**2

                # Gradiente salida
                delta_out = error * self.tita_prime(fwd["output_net"])

                # Gradiente oculta
                delta_hidden = [
                    delta_out * self.w_hidden_output[i + 1] * self.tita_prime(fwd["hidden_net"][i]) for i in range(len(fwd["hidden_net"]))
                ]

                # Actualizar pesos salida
                for i in range(len(self.w_hidden_output)):
                    self.w_hidden_output[i] += self.learning_rate * delta_out * fwd["hidden_out_with_bias"][i]

                # Actualizar pesos entrada -> oculta
                for j in range(len(self.w_input_hidden)):
                    for i in range(len(self.w_input_hidden[j])):
                        self.w_input_hidden[j][i] += self.learning_rate * delta_hidden[j] * fwd["input"][i]

            history.append(total_error)
            if total_error < 1e-3:
                break

        return history
