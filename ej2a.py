import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

from src.perceptron import PerceptronLineal, PerceptronNoLineal


def sigmoid(x, beta=1):
    return 1 / (1 + math.exp(-2 * beta * x))


def sigmoid_derivative(x, beta=1):
    s = sigmoid(x, beta)
    return 2 * beta * s * (1 - s)


def tanh(x):
    return math.tanh(x)


def tanh_derivative(x):
    return 1 - math.tanh(x) ** 2


def prepare_data(raw_x):
    return np.array([[1] + list(row) for row in raw_x])


def plot_prediction_vs_real(data, labels, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, "rs-", label="Predicho")
    plt.plot(labels, "bo:", label="Real")
    plt.title("Predicción vs Real")
    plt.xlabel("Índice de muestra")
    plt.ylabel("Valor de salida")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_training_error(timelapse: dict):
    epochs = sorted(int(k) for k in timelapse["lapse"].keys())
    errors = [timelapse["lapse"][epoch]["total_error"] for epoch in epochs]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, errors, marker="o", linestyle="-", color="purple")
    plt.xlabel("Época")
    plt.ylabel("Error total")
    plt.title("Error total durante el entrenamiento")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def get_activation_function(name):
    if name == "sigmoid":
        return sigmoid, sigmoid_derivative
    elif name == "tanh":
        return tanh, tanh_derivative
    else:
        raise ValueError(f"Unknown activation function: {name}")


def main():
    parser = argparse.ArgumentParser(description="Train a perceptron on the dataset.")
    parser.add_argument("--type", choices=["linear", "nonlinear"], required=True, help="Type of perceptron to use (linear or nonlinear)")
    parser.add_argument("--activation", choices=["sigmoid", "tanh"], default="sigmoid", help="Activation function (default: sigmoid)")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for training (default: 0.001)")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs for training (default: 10000)")

    args = parser.parse_args()

    data = np.loadtxt("assets/conjunto.csv", delimiter=",", skiprows=1)
    x = data[:, :-1]
    y = data[:, -1]

    data = prepare_data(x)
    min_y = min(y)
    max_y = max(y)
    y = np.array([(yi - min_y) / (max_y - min_y) for yi in y])

    if args.type == "linear":
        perceptron = PerceptronLineal(input_size=3, learning_rate=args.learning_rate)
        timelapse = perceptron.train(data, y, epochs=args.epochs)
        predictions = np.array([perceptron.predict(xi) for xi in data])
        output_file = f"results/ej2a/timelapse_ej2_lineal_lr{args.learning_rate}.json"
    else:
        tita, tita_prime = get_activation_function(args.activation)
        perceptron = PerceptronNoLineal(input_size=3, learning_rate=args.learning_rate, tita=tita, tita_prime=tita_prime)
        timelapse = perceptron.train(data, y, epochs=args.epochs)
        predictions = np.array([perceptron.predict(xi)[0] for xi in data])
        output_file = f"results/ej2a/timelapse_ej2_nolineal_lr{args.learning_rate}_{args.activation}.json"

    print(f"Entrenamiento completado en la época {len(timelapse['lapse'])}")
    print("Predicciones finales:")

    for i, (xi, yi) in enumerate(zip(data, y)):
        y_hat = predictions[i]
        print(f"- Input: ({xi[1]:5.2f}, {xi[2]:5.2f}, {xi[3]:5.2f}) -> Predicción: {y_hat:5.2f}, Esperado: {yi:5.2f}")

    plot_prediction_vs_real(data, y, predictions)
    plot_training_error(timelapse)

    os.makedirs("results/ej2a", exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(timelapse, f, indent=2)


if __name__ == "__main__":
    main()
