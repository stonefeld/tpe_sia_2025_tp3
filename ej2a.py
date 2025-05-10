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


def plot_errors(lineal_errors, nolineal_errors):
    lineal_labels = lineal_errors.keys()
    nolineal_labels = nolineal_errors.keys()

    lineal_avg_errors = [x["average"] for x in lineal_errors.values()]
    nolineal_avg_errors = [x["average"] for x in nolineal_errors.values()]

    lineal_std_errors = [x["std"] for x in lineal_errors.values()]
    nolineal_std_errors = [x["std"] for x in nolineal_errors.values()]

    min_lineal = min(lineal_errors.items(), key=lambda x: x[1]["average"])
    min_nolineal = min(nolineal_errors.items(), key=lambda x: x[1]["average"])

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.errorbar(lineal_labels, lineal_avg_errors, yerr=lineal_std_errors, marker="o", linestyle="-", color="purple", label="Lineal")
    ax.errorbar(nolineal_labels, nolineal_avg_errors, yerr=nolineal_std_errors, marker="o", linestyle="-", color="orange", label="No Lineal")
    ax.plot(min_lineal[0], min_lineal[1]["average"], marker="*", color="red", markersize=15)
    ax.plot(min_nolineal[0], min_nolineal[1]["average"], marker="*", color="red", markersize=15)

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Error total")
    ax.set_title("Error total vs Learning Rate")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.savefig("error_lineal_nolineal.png")

    plt.tight_layout()
    plt.show()

    return min_lineal[0], min_nolineal[0]


def plot_prediction_vs_real(labels, predictions_lineal, predictions_nolineal):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(predictions_lineal, "rs-", label="Predicho (lineal)")
    ax.plot(predictions_nolineal, "gx-", label="Predicho (no lineal)")
    ax.plot(labels, "bo:", label="Real")

    ax.set_title("Predicción (Lineal vs No Lineal) vs Real")
    ax.set_xlabel("Índice de muestra")
    ax.set_ylabel("Valor de salida")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.savefig("prediction_vs_real.png")

    plt.tight_layout()
    plt.show()


def plot_training_error(timelapse_lineal, timelapse_nolineal):
    epochs_lineal = sorted(int(k) for k in timelapse_lineal["lapse"].keys())
    errors_lineal = [timelapse_lineal["lapse"][epoch]["total_error"] for epoch in epochs_lineal]

    epochs_nolineal = sorted(int(k) for k in timelapse_nolineal["lapse"].keys())
    errors_nolineal = [timelapse_nolineal["lapse"][epoch]["total_error"] for epoch in epochs_nolineal]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epochs_lineal, errors_lineal, marker="o", linestyle="-", color="purple", label="Lineal")
    ax.plot(epochs_nolineal, errors_nolineal, marker="o", linestyle="-", color="orange", label="No Lineal")

    ax.set_xlabel("Época")
    ax.set_ylabel("Error total")
    ax.set_title("Error total durante el entrenamiento (Lineal vs No Lineal)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.savefig("training_error.png")

    plt.tight_layout()
    plt.show()


def plot_average_error(lineal_average_errors, nolineal_average_errors):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(["Lineal", "No Lineal"], [lineal_average_errors, nolineal_average_errors])
    ax.set_xlabel("Perceptron")
    ax.set_ylabel("Error promedio")
    ax.set_title("Error promedio de cada perceptron")
    fig.savefig("average_error.png")

    plt.tight_layout()
    plt.show()


def main():
    conjunto = np.loadtxt("assets/conjunto.csv", delimiter=",", skiprows=1)
    x = conjunto[:, :-1]
    y = conjunto[:, -1]

    learning_rates = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    repeats = 10

    min_y = min(y)
    max_y = max(y)
    y = np.array([(i - min_y) / (max_y - min_y) for i in y])

    lineal_errors = {}
    nolineal_errors = {}
    lineal_predict = {}
    nolineal_predict = {}
    lineal_timelapse = {}
    nolineal_timelapse = {}

    for learning_rate in learning_rates:
        lineal_errors[learning_rate] = {"errors": [], "std": 0, "average": 0}
        nolineal_errors[learning_rate] = {"errors": [], "std": 0, "average": 0}
        lineal_predict[learning_rate] = []
        nolineal_predict[learning_rate] = []
        lineal_timelapse[learning_rate] = {}
        nolineal_timelapse[learning_rate] = {}

        for _ in range(repeats):
            plineal = PerceptronLineal(input_size=x.shape[1], learning_rate=learning_rate)

            timelapse_lineal = plineal.train(x, y, epochs=1000)
            last_epoch = max(int(k) for k in timelapse_lineal["lapse"].keys())
            lineal_errors[learning_rate]["errors"].append(timelapse_lineal["lapse"][last_epoch]["total_error"])

            if len(lineal_predict[learning_rate]) == 0:
                lineal_predict[learning_rate] = [plineal.predict(i) for i in x]
                lineal_timelapse[learning_rate] = timelapse_lineal

            # NO LINEAL
            pnolineal = PerceptronNoLineal(input_size=x.shape[1], learning_rate=learning_rate, tita=sigmoid, tita_prime=sigmoid_derivative)

            timelapse_nolineal = pnolineal.train(x, y, epochs=1000)
            last_epoch = max(int(k) for k in timelapse_nolineal["lapse"].keys())
            nolineal_errors[learning_rate]["errors"].append(timelapse_nolineal["lapse"][last_epoch]["total_error"])

            if len(nolineal_predict[learning_rate]) == 0:
                nolineal_predict[learning_rate] = [pnolineal.predict(i)[0] for i in x]
                nolineal_timelapse[learning_rate] = timelapse_nolineal

        lineal_errors[learning_rate]["std"] = np.std(lineal_errors[learning_rate]["errors"])
        lineal_errors[learning_rate]["average"] = np.mean(lineal_errors[learning_rate]["errors"])
        nolineal_errors[learning_rate]["std"] = np.std(nolineal_errors[learning_rate]["errors"])
        nolineal_errors[learning_rate]["average"] = np.mean(nolineal_errors[learning_rate]["errors"])

    best_lineal, best_nolineal = plot_errors(lineal_errors, nolineal_errors)
    plot_prediction_vs_real(y, lineal_predict[best_lineal], nolineal_predict[best_nolineal])
    plot_training_error(lineal_timelapse[best_lineal], nolineal_timelapse[best_nolineal])
    plot_average_error(lineal_errors[best_lineal]["average"], nolineal_errors[best_nolineal]["average"])


if __name__ == "__main__":
    main()
