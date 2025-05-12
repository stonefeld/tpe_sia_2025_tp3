import csv
import math
import random

import matplotlib.pyplot as plt
import numpy as np

from src.perceptron import PerceptronMulticapa


def tanh(h):
    return math.tanh(h)


def tanh_prime(h):
    return 1 - math.tanh(h) ** 2


def cargar_digitos_y_etiquetas(path="assets/digitos.txt"):
    with open(path) as f:
        lines = [list(map(int, line.strip().split())) for line in f if line.strip()]

    digitos = []
    for i in range(0, len(lines), 7):
        bloque = lines[i : i + 7]
        flatten = [bit for fila in bloque for bit in fila]
        digitos.append(flatten)

    etiquetas = np.array([1 if i % 2 == 0 else -1 for i in range(10)])
    return np.array(digitos), etiquetas


def calcular_metricas_binarias(predichos, esperados):
    TP = TN = FP = FN = 0
    for pred, esp in zip(predichos, esperados):
        if esp == 1 and pred == 1:
            TP += 1
        elif esp == -1 and pred == -1:
            TN += 1
        elif esp == -1 and pred == 1:
            FP += 1
        elif esp == 1 and pred == -1:
            FN += 1

    total = TP + TN + FP + FN
    if total == 0:
        return 0, 0

    accuracy = (TP + TN) / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    return accuracy, precision


def k_fold_cross_validation(data, labels, k=10, epocas=300):
    combined = list(zip(data, labels))
    # random.shuffle(combined)

    fold_size = len(data) // k
    fold_errors = []
    fold_preds = []

    for fold in range(k):
        iter_preds = []

        for _ in range(20):
            # Separar datos
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

            # Inicializar y entrenar perceptrón
            mlp = PerceptronMulticapa(capas=[data.shape[1], 10, 1], tita=tanh, tita_prime=tanh_prime)

            # Entrenar y evaluar
            mlp.train(x_train, y_train, epocas=epocas, tolerancia=0)

            # Evaluar en test
            test_preds = [mlp.predict(x)[0] for x in x_test]
            iter_preds.append(np.mean(test_preds))
            fold_error = np.mean([abs(x - xt) for x, xt in zip(test_preds, y_test)])

        fold_preds.append(iter_preds)
        fold_errors.append(fold_error)

    return fold_preds, fold_errors


def plot_metrics(train_metrics, test_metrics, metric_name, filename):
    plt.figure(figsize=(10, 6))
    folds = range(1, len(train_metrics) + 1)

    plt.plot(folds, train_metrics, "b-", label="Training", marker="o")
    plt.plot(folds, test_metrics, "r-", label="Testing", marker="s")

    plt.xlabel("Fold")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} por Fold")
    plt.xticks(folds)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()


def plot_errors(fold_errors, filename):
    plt.figure(figsize=(10, 6))
    folds = range(len(fold_errors))
    avg_error = np.mean(fold_errors)

    plt.bar(folds, fold_errors, color="skyblue", alpha=0.7)
    plt.axhline(y=avg_error, color="red", linestyle="--", label=f"Error promedio: {avg_error:.4f}")

    plt.xlabel("Número sacado del conjunto")
    plt.ylabel("Error")
    plt.title("Error por número sacado del conjunto")
    plt.xticks(folds)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.show()


def plot_predictions(predictions, filename):
    plt.figure(figsize=(10, 6))
    plt.boxplot(predictions, labels=range(0, len(predictions)))
    plt.xlabel("Número sacado del conjunto")
    plt.ylabel("Predicción")
    plt.title("Predicciones por número sacado del conjunto")
    plt.savefig(filename)
    plt.show()


def main():
    data, labels = cargar_digitos_y_etiquetas()

    fold_preds, fold_errors = k_fold_cross_validation(data, labels, k=10, epocas=1000)

    plot_predictions(fold_preds, "predictions_vs_fold.png")
    plot_errors(fold_errors, "errors_vs_fold.png")

if __name__ == "__main__":
    main()
