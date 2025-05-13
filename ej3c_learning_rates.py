import math
import os
import re
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.optimizers import SGD, Adam, Momentum
from src.perceptron import PerceptronMulticapa


def tanh(h):
    return math.tanh(h)


def tanh_prime(salida):
    return 1 - salida**2


def cargar_imagen_como_vector(path_imagen):
    with Image.open(path_imagen) as img:
        img = img.convert("L")
        pixeles = np.array(img.getdata())
        binarizado = np.where(pixeles < 128, 1, 0)
        return binarizado


def cargar_imagenes_y_etiquetas(carpeta):
    imagenes = []
    etiquetas = []

    archivos = [f for f in os.listdir(carpeta) if f.endswith(".png")]
    archivos.sort()

    patron = re.compile(r"imagen_(\d+)_\w+\.png")

    min_numero = float("inf")
    max_numero = float("-inf")
    for archivo in archivos:
        match = patron.match(archivo)
        if match:
            numero = int(match.group(1))
            min_numero = min(min_numero, numero)
            max_numero = max(max_numero, numero)

    num_outputs = max_numero - min_numero + 1

    for archivo in archivos:
        path = os.path.join(carpeta, archivo)
        vector = cargar_imagen_como_vector(path)

        match = patron.match(archivo)
        if not match:
            continue

        numero = int(match.group(1))
        etiqueta = np.full(num_outputs, -1)
        etiqueta[numero - min_numero] = 1

        imagenes.append(vector)
        etiquetas.append(etiqueta)

    return np.array(imagenes), np.array(etiquetas), num_outputs


def calcular_metricas(preds, targets):
    clases = np.unique([np.argmax(t) for t in targets])
    tp = {c: 0 for c in clases}
    fp = {c: 0 for c in clases}
    correct = 0

    for p, t in zip(preds, targets):
        pred_idx = np.argmax(p)
        true_idx = np.argmax(t)
        if pred_idx == true_idx:
            correct += 1
            tp[pred_idx] += 1
        else:
            fp[pred_idx] += 1

    accuracy = correct / len(targets)

    precision_por_clase = []
    for c in clases:
        denom = tp[c] + fp[c]
        precision_c = tp[c] / denom if denom > 0 else 0
        precision_por_clase.append(precision_c)

    precision_macro = sum(precision_por_clase) / len(clases)
    return accuracy, precision_macro


def plot_learning_rate_errors(optimizer_errors):
    fig, ax = plt.subplots(figsize=(10, 5))

    for optimizer_name, errors in optimizer_errors.items():
        learning_rates = list(errors.keys())
        avg_errors = [x["average"] for x in errors.values()]
        std_errors = [x["std"] for x in errors.values()]

        min_error = min(errors.items(), key=lambda x: x[1]["average"])
        
        ax.errorbar(learning_rates, avg_errors, yerr=std_errors, marker="o", linestyle="-", label=optimizer_name)
        ax.plot(min_error[0], min_error[1]["average"], marker="*", color="red", markersize=15)

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Error total")
    ax.set_title("Error total vs Learning Rate por Optimizador")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.savefig("learning_rate_errors.png")

    plt.tight_layout()
    plt.show()


def plot_training_curves(optimizer_curves):
    fig, ax = plt.subplots(figsize=(10, 5))

    for optimizer_name, curve in optimizer_curves.items():
        epochs = range(1, len(curve) + 1)
        ax.plot(epochs, curve, marker="o", linestyle="-", label=optimizer_name)

    ax.set_xlabel("Época")
    ax.set_ylabel("Error total")
    ax.set_title("Error total durante el entrenamiento por Optimizador")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.savefig("training_curves.png")

    plt.tight_layout()
    plt.show()


def main():
    X_train, Y_train, num_outputs = cargar_imagenes_y_etiquetas("assets/training_set")
    X_test, Y_test, _ = cargar_imagenes_y_etiquetas("assets/testing_set")

    input_size = len(X_train[0])
    capas = [input_size, 30, num_outputs]

    learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
    repeats = 1
    epocas = 300

    optimizer_errors = {
        "SGD": {},
        "Momentum": {},
        "Adam": {}
    }

    best_learning_rates = {}
    best_mlps = {}

    # Test different learning rates for each optimizer
    for lr in learning_rates:
        for optimizer_name in optimizer_errors.keys():
            optimizer_errors[optimizer_name][lr] = {"errors": [], "std": 0, "average": 0}
            
            for _ in range(repeats):
                if optimizer_name == "SGD":
                    optimizer = SGD(learning_rate=lr)
                elif optimizer_name == "Momentum":
                    optimizer = Momentum(learning_rate=lr, momentum=0.8)
                else:  # Adam
                    optimizer = Adam(learning_rate=lr, layers=capas)

                mlp = PerceptronMulticapa(capas, tita=tanh, tita_prime=tanh_prime, optimizer=optimizer)
                results = mlp.train(X_train, Y_train, epocas=epocas, tolerancia=0.001)
                optimizer_errors[optimizer_name][lr]["errors"].append(results["errors"][-1])

            optimizer_errors[optimizer_name][lr]["std"] = np.std(optimizer_errors[optimizer_name][lr]["errors"])
            optimizer_errors[optimizer_name][lr]["average"] = np.mean(optimizer_errors[optimizer_name][lr]["errors"])

    # Plot learning rate errors
    plot_learning_rate_errors(optimizer_errors)

    # Find best learning rate for each optimizer
    for optimizer_name in optimizer_errors.keys():
        best_lr = min(optimizer_errors[optimizer_name].items(), key=lambda x: x[1]["average"])[0]
        best_learning_rates[optimizer_name] = best_lr

        # Train final model with best learning rate
        if optimizer_name == "SGD":
            optimizer = SGD(learning_rate=best_lr)
        elif optimizer_name == "Momentum":
            optimizer = Momentum(learning_rate=best_lr, momentum=0.8)
        else:  # Adam
            optimizer = Adam(learning_rate=best_lr, layers=capas)

        mlp = PerceptronMulticapa(capas, tita=tanh, tita_prime=tanh_prime, optimizer=optimizer)
        results = mlp.train(X_train, Y_train, epocas=epocas, tolerancia=0.001)
        best_mlps[optimizer_name] = mlp

        # Save results to CSV
        with open(f"resultados_test_{optimizer_name.lower()}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Imagen", "Esperado", "Predicho", "Salidas"])
            for i, (x, t) in enumerate(zip(X_test, Y_test)):
                salida = mlp.forward(x)[-1]
                predicho = np.argmax(salida)
                esperado = np.argmax(t)
                writer.writerow([i, esperado, predicho, ",".join(f"{s:.5f}" for s in salida)])

        # Calculate and print metrics
        pred_train = [mlp.forward(x)[-1] for x in X_train]
        acc_train, prec_train = calcular_metricas(pred_train, Y_train)
        
        pred_test = [mlp.forward(x)[-1] for x in X_test]
        acc_test, prec_test = calcular_metricas(pred_test, Y_test)

        print(f"\nResultados para {optimizer_name} (lr={best_lr}):")
        print(f"Train - Accuracy: {acc_train:.4f}, Precision: {prec_train:.4f}")
        print(f"Test  - Accuracy: {acc_test:.4f}, Precision: {prec_test:.4f}")

        # Save metrics to CSV
        with open(f"accuracy_precision_{optimizer_name.lower()}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_accuracy", "train_precision", "test_accuracy", "test_precision"])
            for i in range(epocas):
                writer.writerow([
                    i + 1,
                    acc_train,
                    prec_train,
                    acc_test,
                    prec_test
                ])


if __name__ == "__main__":
    main() 