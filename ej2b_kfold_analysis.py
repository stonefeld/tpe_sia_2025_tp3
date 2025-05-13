import math
import random

import matplotlib.pyplot as plt
import numpy as np

from ej2b import sigmoid, sigmoid_derivative
from src.perceptron import PerceptronNoLineal


def k_fold_cross_validation_nolineal(data, labels, k, tita, tita_prime, learning_rate=0.01, epochs=1000):
    combined = list(zip(data, labels))
    random.shuffle(combined)

    fold_size = len(data) // k
    train_errors = []
    test_errors = []

    for fold in range(k):
        left = fold * fold_size
        right = (fold + 1) * fold_size
        test_data = combined[left:right]
        train_data = combined[:left] + combined[right:]

        x_train, y_train = zip(*train_data)
        x_test, y_test = zip(*test_data)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        p = PerceptronNoLineal(input_size=len(x_train[0]), tita=tita, tita_prime=tita_prime, learning_rate=learning_rate)
        results = p.train(x_train, y_train, epochs, tolerance=1e-3)

        train_predictions = [p.predict(xi)[0] for xi in x_train]
        train_error = np.mean(np.abs(np.array(y_train) - np.array(train_predictions)))

        test_predictions = [p.predict(xi)[0] for xi in x_test]
        test_error = np.mean(np.abs(np.array(y_test) - np.array(test_predictions)))

        train_errors.append(train_error)
        test_errors.append(test_error)

    return train_errors, test_errors


def main():
    conjunto = np.loadtxt("assets/conjunto.csv", delimiter=",", skiprows=1)
    x = conjunto[:, :-1]
    y = conjunto[:, -1]

    # Normalizar salidas
    min_y = min(y)
    max_y = max(y)
    y = np.array([(i - min_y) / (max_y - min_y) for i in y])

    # Valores de K a probar
    ks = [2, 4, 7, 14, 28]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    x_pos = np.arange(len(ks))
    width = 0.35

    train_errors = []
    test_errors = []

    for k in ks:
        print(f"Evaluando k = {k}")
        train_error, test_error = k_fold_cross_validation_nolineal(
            x,
            y,
            k=k,
            tita=sigmoid,
            tita_prime=sigmoid_derivative,
            learning_rate=5e-2,
            epochs=100,
        )
        train_errors.append(np.mean(train_error))
        test_errors.append(np.mean(test_error))

    ax.bar(x_pos - width / 2, train_errors, width, label="Entrenamiento", color="orange")
    ax.bar(x_pos + width / 2, test_errors, width, label="Testeo", color="blue")

    ax.set_ylabel("Error promedio")
    ax.set_xlabel("Valor de k")
    ax.set_title("Comparación de errores de entrenamiento y testeo")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ks)
    ax.legend()

    for i, v in enumerate(train_errors):
        ax.text(i - width / 2, v, f"{v:.4f}", ha="center", va="bottom")
    for i, v in enumerate(test_errors):
        ax.text(i + width / 2, v, f"{v:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.savefig("grafico_error_vs_k_ej2b.png")
    plt.show()


if __name__ == "__main__":
    main()
