import csv
import json
import math

import matplotlib.pyplot as plt

from src.perceptron import PerceptronLineal, PerceptronNoLineal


def sigmoid(x, beta=1):
    return 1 / (1 + math.exp(-2 * beta * x))


def sigmoid_derivative(x, beta=1):
    s = sigmoid(x, beta)
    return 2 * beta * s * (1 - s)


def prepare_data(raw_x):
    return [[1] + list(row) for row in raw_x]


def plot_prediction_vs_real(data, labels, predictions_lineal, predictions_nolineal):
    plt.figure(figsize=(10, 6))
    plt.plot(predictions_lineal, "rs-", label="Predicho (lineal)")
    plt.plot(predictions_nolineal, "gx-", label="Predicho (no lineal)")
    plt.plot(labels, "bo:", label="Real")
    plt.title("Predicción (Lineal vs No Lineal) vs Real")
    plt.xlabel("Índice de muestra")
    plt.ylabel("Valor de salida")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_training_error(timelapse_lineal: dict, timelapse_nolineal: dict):
    epochs_lineal = sorted(int(k) for k in timelapse_lineal["lapse"].keys())
    errors_lineal = [timelapse_lineal["lapse"][epoch]["total_error"] for epoch in epochs_lineal]

    epochs_nolineal = sorted(int(k) for k in timelapse_nolineal["lapse"].keys())
    errors_nolineal = [timelapse_nolineal["lapse"][epoch]["total_error"] for epoch in epochs_nolineal]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_lineal, errors_lineal, marker="o", linestyle="-", color="purple", label="Lineal")
    plt.plot(epochs_nolineal, errors_nolineal, marker="o", linestyle="-", color="orange", label="No Lineal")
    plt.xlabel("Época")
    plt.ylabel("Error total")
    plt.title("Error total durante el entrenamiento (Lineal vs No Lineal)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    with open("assets/conjunto.csv", "r") as f:
        lines = csv.reader(f)
        data = list(lines)[1:]  # Skip header
        x = [list(map(float, row[:-1])) for row in data]
        y = [float(row[-1]) for row in data]

    data = prepare_data(x)
    max_y = max(abs(i) for i in y)
    y = [i / max_y for i in y]
    plineal = PerceptronLineal(input_size=3, learning_rate=0.001)

    timelapse_lineal = plineal.train(data, y)
    predictions_lineal = [plineal.predict(xi) for xi in data]

    print(f"Entrenamiento completado en la época {len(timelapse_lineal['lapse'])}")
    print("Predicciones finales:")
    for i, (xi, yi) in enumerate(zip(data, y)):
        y_hat = predictions_lineal[i]
        print(f"- Input: ({xi[1]}, {xi[2]}, {xi[3]}) -> Predicción: {y_hat}, Esperado: {yi}")

    with open("timelapse_ej2_lineal.json", "w") as f:
        json.dump(timelapse_lineal, f, indent=2)

    # NO LINEAL
    pnolineal = PerceptronNoLineal(input_size=3, learning_rate=0.00001, tita=sigmoid, tita_prime=sigmoid_derivative)

    timelapse_nolineal = pnolineal.train(data, y, epochs=1000000)
    predictions_nolineal = [pnolineal.predict(xi)[0] for xi in data]

    plot_prediction_vs_real(data, y, predictions_lineal, predictions_nolineal)
    plot_training_error(timelapse_lineal, timelapse_nolineal)

    print(f"Entrenamiento completado en la época {len(timelapse_nolineal['lapse'])}")
    print("Predicciones finales:")
    for i, (xi, yi) in enumerate(zip(data, y)):
        y_hat = predictions_nolineal[i]
        print(f"- Input: ({xi[1]}, {xi[2]}, {xi[3]}) -> Predicción: {y_hat}, Esperado: {yi}")

    with open("timelapse_ej2_nolineal.json", "w") as f:
        json.dump(timelapse_lineal, f, indent=2)


if __name__ == "__main__":
    main()
