import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.perceptron import PerceptronLineal, PerceptronNoLineal


def sigmoid(x, beta=1):
    return 1 / (1 + math.exp(-2 * beta * x))


def sigmoid_derivative(x, beta=1):
    s = sigmoid(x, beta)
    return 2 * beta * s * (1 - s)


def prepare_data(raw_x):
    return np.array([[1] + list(row) for row in raw_x])


def run_multiple_trainings(
    data: np.ndarray,
    labels: np.ndarray,
    learning_rates: List[float],
    n_runs: int = 10,
    epochs: int = 1000
) -> Tuple[Dict[float, List[float]], Dict[float, List[float]]]:
    """Run multiple training sessions for each learning rate and return average errors."""
    lineal_errors = {lr: [] for lr in learning_rates}
    nolineal_errors = {lr: [] for lr in learning_rates}

    for lr in learning_rates:
        for _ in range(n_runs):
            # Lineal
            plineal = PerceptronLineal(input_size=len(data[0]) - 1, learning_rate=lr)
            timelapse_lineal = plineal.train(data, labels, epochs=epochs)
            last_epoch = max(int(k) for k in timelapse_lineal["lapse"].keys())
            final_error = timelapse_lineal["lapse"][last_epoch]["total_error"]
            lineal_errors[lr].append(final_error)

            # No Lineal
            pnolineal = PerceptronNoLineal(input_size=len(data[0]) - 1, learning_rate=lr, tita=sigmoid, tita_prime=sigmoid_derivative)
            timelapse_nolineal = pnolineal.train(data, labels, epochs=epochs)
            last_epoch = max(int(k) for k in timelapse_nolineal["lapse"].keys())
            final_error = timelapse_nolineal["lapse"][last_epoch]["total_error"]
            nolineal_errors[lr].append(final_error)

    return lineal_errors, nolineal_errors


def plot_error_vs_learning_rate(
    lineal_errors: Dict[float, List[float]],
    nolineal_errors: Dict[float, List[float]]
):
    """Plot average error vs learning rate with standard deviation."""
    lrs = sorted(lineal_errors.keys())
    lineal_means = [np.mean(lineal_errors[lr]) for lr in lrs]
    lineal_stds = [np.std(lineal_errors[lr]) for lr in lrs]
    nolineal_means = [np.mean(nolineal_errors[lr]) for lr in lrs]
    nolineal_stds = [np.std(nolineal_errors[lr]) for lr in lrs]

    plt.figure(figsize=(12, 6))
    
    # Lineal
    plt.errorbar(lrs, lineal_means, yerr=lineal_stds, fmt='o-', label='Lineal', color='blue')
    min_lineal_idx = np.argmin(lineal_means)
    plt.plot(lrs[min_lineal_idx], lineal_means[min_lineal_idx], 'r*', markersize=15, 
             label=f'Min Lineal: {lineal_means[min_lineal_idx]:.4f}')

    # No Lineal
    plt.errorbar(lrs, nolineal_means, yerr=nolineal_stds, fmt='s-', label='No Lineal', color='green')
    min_nolineal_idx = np.argmin(nolineal_means)
    plt.plot(lrs[min_nolineal_idx], nolineal_means[min_nolineal_idx], 'r*', markersize=15,
             label=f'Min No Lineal: {nolineal_means[min_nolineal_idx]:.4f}')

    plt.xlabel('Learning Rate')
    plt.ylabel('Average Error')
    plt.title('Error vs Learning Rate (with std)')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.tight_layout()
    plt.show()

    return lrs[min_lineal_idx], lrs[min_nolineal_idx]


def plot_final_comparison(
    data: np.ndarray,
    labels: np.ndarray,
    best_lr_lineal: float,
    best_lr_nolineal: float,
    epochs: int = 1000
):
    """Train with best learning rates and create comparison plots."""
    # Train with best learning rates
    plineal = PerceptronLineal(input_size=3, learning_rate=best_lr_lineal)
    timelapse_lineal = plineal.train(data, labels, epochs=epochs)
    predictions_lineal = np.array([plineal.predict(xi) for xi in data])

    pnolineal = PerceptronNoLineal(input_size=3, learning_rate=best_lr_nolineal, tita=sigmoid, tita_prime=sigmoid_derivative)
    timelapse_nolineal = pnolineal.train(data, labels, epochs=epochs)
    predictions_nolineal = np.array([pnolineal.predict(xi)[0] for xi in data])

    # Plot predictions vs real
    plt.figure(figsize=(12, 6))
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

    # Plot error vs epoch
    epochs_lineal = sorted(int(k) for k in timelapse_lineal["lapse"].keys())
    errors_lineal = [timelapse_lineal["lapse"][epoch]["total_error"] for epoch in epochs_lineal]

    epochs_nolineal = sorted(int(k) for k in timelapse_nolineal["lapse"].keys())
    errors_nolineal = [timelapse_nolineal["lapse"][epoch]["total_error"] for epoch in epochs_nolineal]

    plt.figure(figsize=(12, 6))
    plt.plot(epochs_lineal, errors_lineal, marker="o", linestyle="-", color="purple", label="Lineal")
    plt.plot(epochs_nolineal, errors_nolineal, marker="o", linestyle="-", color="orange", label="No Lineal")
    plt.xlabel("Época")
    plt.ylabel("Error total")
    plt.title("Error total durante el entrenamiento (Lineal vs No Lineal)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot average error bar plot
    plt.figure(figsize=(8, 6))
    errors = [np.mean(errors_lineal), np.mean(errors_nolineal)]
    stds = [np.std(errors_lineal), np.std(errors_nolineal)]
    plt.bar(['Lineal', 'No Lineal'], errors, yerr=stds, capsize=10)
    plt.ylabel('Error promedio')
    plt.title('Comparación de error promedio entre perceptrones')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


def main():
    # Load and prepare data
    data = np.loadtxt("assets/conjunto.csv", delimiter=",", skiprows=1)
    x = data[:, :-1]
    y = data[:, -1]

    data = prepare_data(x)
    max_y = max(abs(i) for i in y)
    y = np.array([i / max_y for i in y])

    # Define learning rates to test
    learning_rates = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.0125]

    # Run multiple trainings and get errors
    lineal_errors, nolineal_errors = run_multiple_trainings(data, y, learning_rates)

    # Plot error vs learning rate and get best learning rates
    best_lr_lineal, best_lr_nolineal = plot_error_vs_learning_rate(lineal_errors, nolineal_errors)

    # Train with best learning rates and create comparison plots
    plot_final_comparison(data, y, best_lr_lineal, best_lr_nolineal)


if __name__ == "__main__":
    main() 