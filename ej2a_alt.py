import csv
import json
import math

import matplotlib.pyplot as plt
import numpy as np

from src.perceptron import PerceptronLineal, PerceptronNoLineal


def sigmoid(x, beta=1):
    return 1 / (1 + math.exp(-2 * beta * x))


def sigmoid_derivative(x, beta=1):
    s = sigmoid(x, beta)
    return 2 * beta * s * (1 - s)


def normalize_data(x, y):
    # normaliza x por min-max feature-wise a [0,1]
    x_t = list(zip(*x))  # transponer
    x_norm = []
    for feature in x_t:
        min_f, max_f = min(feature), max(feature)
        if max_f - min_f == 0:
            x_norm.append([0.0 for _ in feature])
        else:
            x_norm.append([(v - min_f) / (max_f - min_f) for v in feature])
    x_norm = list(zip(*x_norm))

    # normalizar y
    min_y, max_y = min(y), max(y)
    y_norm = [(v - min_y) / (max_y - min_y) for v in y]

    return x_norm, y_norm, (min_y, max_y)


def prepare_data(raw_x):
    return [[1] + list(row) for row in raw_x]


def benchmark_perceptrones(data, y, tita, tita_prime, learning_rates=[0.001, 0.01, 0.05], runs=5, epochs=1000):
    resultados = {"lineal": {}, "nolineal": {}}

    for lr in learning_rates:
        errores_lineal = []
        epocas_lineal = []

        errores_nolineal = []
        epocas_nolineal = []

        for _ in range(runs):
            # LINEAL
            pl = PerceptronLineal(input_size=len(data[0]) - 1, learning_rate=lr)
            tl_l = pl.train(data, y, epochs=epochs)
            err_final = tl_l["lapse"][list(tl_l["lapse"].keys())[-1]]["total_error"]
            errores_lineal.append(err_final)
            epocas_lineal.append(len(tl_l["lapse"]))

            # NO LINEAL
            pnl = PerceptronNoLineal(input_size=len(data[0]) - 1, learning_rate=lr, tita=tita, tita_prime=tita_prime)
            tl_nl = pnl.train(data, y, epochs=epochs)
            err_final_nl = tl_nl["lapse"][list(tl_nl["lapse"].keys())[-1]]["total_error"]
            errores_nolineal.append(err_final_nl)
            epocas_nolineal.append(len(tl_nl["lapse"]))

            resultados["lineal"][lr] = {
                "error_promedio": np.mean(errores_lineal),
                "error_std": np.std(errores_lineal) if len(errores_lineal) > 1 else 0,
                "epocas_promedio": np.mean(epocas_lineal),
                "epocas_std": np.std(epocas_lineal) if len(epocas_lineal) > 1 else 0,
            }

            resultados["nolineal"][lr] = {
                "error_promedio": np.mean(errores_nolineal),
                "error_std": np.std(errores_nolineal),
                "epocas_promedio": np.mean(epocas_nolineal),
                "epocas_std": np.std(epocas_nolineal),
            }

    plot_benchmark(resultados)
    return resultados


def plot_benchmark(resultados):
    learning_rates = sorted(resultados["lineal"].keys())

    errores_lineal = [resultados["lineal"][lr]["error_promedio"] for lr in learning_rates]
    errores_nolineal = [resultados["nolineal"][lr]["error_promedio"] for lr in learning_rates]

    desvio_lineal = [resultados["lineal"][lr]["error_std"] for lr in learning_rates]
    desvio_nolineal = [resultados["nolineal"][lr]["error_std"] for lr in learning_rates]

    epocas_lineal = [resultados["lineal"][lr]["epocas_promedio"] for lr in learning_rates]
    epocas_nolineal = [resultados["nolineal"][lr]["epocas_promedio"] for lr in learning_rates]

    epocas_std_lineal = [resultados["lineal"][lr]["epocas_std"] for lr in learning_rates]
    epocas_std_nolineal = [resultados["nolineal"][lr]["epocas_std"] for lr in learning_rates]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].errorbar(learning_rates, errores_lineal, yerr=desvio_lineal, fmt="o-", color="purple", label="Lineal", capsize=5)
    axs[0].errorbar(learning_rates, errores_nolineal, yerr=desvio_nolineal, fmt="o-", color="orange", label="No Lineal", capsize=5)
    axs[0].set_title("Error promedio final vs Learning Rate")
    axs[0].set_xlabel("Learning rate")
    axs[0].set_ylabel("Error total promedio")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].errorbar(learning_rates, epocas_lineal, yerr=epocas_std_lineal, fmt="o-", color="purple", label="Lineal", capsize=5)
    axs[1].errorbar(learning_rates, epocas_nolineal, yerr=epocas_std_nolineal, fmt="o-", color="orange", label="No Lineal", capsize=5)
    axs[1].set_title("Épocas promedio hasta convergencia vs Learning Rate")
    axs[1].set_xlabel("Learning rate")
    axs[1].set_ylabel("Épocas promedio")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    with open("assets/conjunto.csv", "r") as f:
        lines = csv.reader(f)
        data = list(lines)[1:]  # Skip header
        x = [list(map(float, row[:-1])) for row in data]
        y = [float(row[-1]) for row in data]

    x_norm, y_norm, y_range = normalize_data(x, y)
    data = prepare_data(x_norm)

    resultados = benchmark_perceptrones(
        data,
        y_norm,
        tita=sigmoid,
        tita_prime=sigmoid_derivative,
        learning_rates=[0.0005, 0.001, 0.005, 0.01, 0.05],
        runs=10,
        epochs=100000,
    )

    with open("benchmark_resultados.json", "w") as f:
        json.dump(resultados, f, indent=2)


if __name__ == "__main__":
    main()
