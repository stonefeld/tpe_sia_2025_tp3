import csv
import math
import random

from src.perceptron import PerceptronNoLineal


def k_fold_cross_validation_nolineal(data, labels, k, tita, tita_prime, learning_rate=0.01, epochs=1000):
    combined = list(zip(data, labels))
    random.shuffle(combined)

    fold_size = len(data) // k
    fold_errors = []

    for fold in range(k):
        # Separar datos
        left = fold * fold_size
        right = (fold + 1) * fold_size
        test_data = combined[left:right]
        train_data = combined[:left] + combined[right:]

        x_train, y_train = zip(*train_data)
        x_test, y_test = zip(*test_data)

        # Inicializar y entrenar perceptrón
        p = PerceptronNoLineal(input_size=len(x_train[0]) - 1, tita=tita, tita_prime=tita_prime, learning_rate=learning_rate)
        p.train(x_train, y_train, epochs)

        # Evaluar en test
        test_predictions = [p.predict(xi)[0] for xi in x_test]
        fold_error = sum(abs(yi - y_hat) for yi, y_hat in zip(y_test, test_predictions)) / len(y_test)
        fold_errors.append(fold_error)

    avg_error = sum(fold_errors) / len(fold_errors)
    return avg_error, fold_errors


def sigmoid(x, beta=1):
    return 1 / (1 + math.exp(-2 * beta * x))


def sigmoid_derivative(x, beta=1):
    s = sigmoid(x, beta)
    return 2 * beta * s * (1 - s)


def prepare_data(raw_x):
    return [[1] + list(row) for row in raw_x]


def main():
    with open("assets/conjunto.csv", "r") as f:
        lines = csv.reader(f)
        data = list(lines)[1:]  # Skip header
        x = [list(map(float, row[:-1])) for row in data]
        y = [float(row[-1]) for row in data]

    data = prepare_data(x)
    max_y = max(abs(i) for i in y)

    y = [i / max_y for i in y]
    avg_error, fold_errors = k_fold_cross_validation_nolineal(
        data, y, k=5, tita=sigmoid, tita_prime=sigmoid_derivative, learning_rate=0.00001, epochs=100000
    )

    print("\n====== VALIDACIÓN CRUZADA (No Lineal) ======")
    for i, err in enumerate(fold_errors):
        print(f"Fold {i+1}: Error promedio = {err:.4f}")
    print(f"Error promedio total (generalización): {avg_error:.4f}")


if __name__ == "__main__":
    main()
