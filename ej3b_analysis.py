import csv
import math

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

    etiquetas = np.array([1 if i % 2 == 1 else -1 for i in range(10)])
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


def main():
    data, labels = cargar_digitos_y_etiquetas()

    mlp = PerceptronMulticapa(capas=[data.shape[1], 10, 1], tita=tanh, tita_prime=tanh_prime)

    epocas = 300
    accs = []
    precs = []

    for epoch in range(epocas):
        mlp.train(data, labels, epocas=1, tolerancia=0.005)

        predichos = [round(mlp.predict(x)[0]) for x in data]
        accuracy, precision = calcular_metricas_binarias(predichos, labels)

        accs.append(accuracy)
        precs.append(precision)

        print(f"Epoch {epoch+1:3} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f}")

    with open("accuracy_precision_vs_epoch_b.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "accuracy", "precision"])
        for i in range(epocas):
            writer.writerow([i + 1, accs[i], precs[i]])

    with open("resultados_digitos.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Dígito", "Predicho", "Esperado", "Paridad", "Salida"])
        for i, x in enumerate(data):
            salida = mlp.predict(x)
            predicho = round(salida[0])
            paridad = "IMPAR" if predicho > 0 else "PAR"
            esperado = labels[i]
            writer.writerow([i, predicho, esperado, paridad, f"{salida[0]:.5f}"])


if __name__ == "__main__":
    main()
