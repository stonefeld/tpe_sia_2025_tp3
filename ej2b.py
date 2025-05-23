import csv
import math
import random

import matplotlib.pyplot as plt
import numpy as np

from src.perceptron import PerceptronNoLineal


def k_fold_cross_validation_nolineal(data, labels, k, tita, tita_prime, learning_rate=0.01, epochs=1000):
    combined = list(zip(data, labels))
    random.shuffle(combined)

    fold_size = len(data) // k
    fold_errors = []

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
        p.train(x_train, y_train, epochs, tolerance=1e-4)

        test_predictions = [p.predict(xi)[0] for xi in x_test]
        fold_error = np.mean(np.abs(np.array(y_test) - np.array(test_predictions)))
        fold_errors.append(fold_error)

    avg_error = np.mean(fold_errors)
    return avg_error, fold_errors


def sigmoid(x, beta=1):
    return 1 / (1 + math.exp(-2 * beta * x))


def sigmoid_derivative(x, beta=1):
    s = sigmoid(x, beta)
    return 2 * beta * s * (1 - s)


def prepare_data(raw_x):
    return np.array([[1] + list(row) for row in raw_x])


def main():
    conjunto = np.loadtxt("assets/conjunto.csv", delimiter=",", skiprows=1)
    x = conjunto[:, :-1]
    y = conjunto[:, -1]

    min_y = min(y)
    max_y = max(y)
    y = np.array([(i - min_y) / (max_y - min_y) for i in y])

    avg_error, fold_errors = k_fold_cross_validation_nolineal(
        x,
        y,
        k=7,
        tita=sigmoid,
        tita_prime=sigmoid_derivative,
        learning_rate=1e-2,
        epochs=10000,
    )

    print("====== VALIDACIÓN CRUZADA (No Lineal) ======")
    for i, err in enumerate(fold_errors):
        print(f"Fold {i+1}: Error promedio = {err:.4f}")
    print(f"Error promedio total (generalización): {avg_error:.4f}")

    plt.figure(figsize=(10, 6))
    folds = range(1, len(fold_errors) + 1)
    plt.bar(folds, fold_errors, color="skyblue", alpha=0.7)
    plt.axhline(y=avg_error, color="red", linestyle="--", label=f"Error promedio: {avg_error:.4f}")

    plt.xlabel("Fold")
    plt.ylabel("Error")
    plt.title("Error por Fold en Validación Cruzada")
    plt.xticks(folds)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
