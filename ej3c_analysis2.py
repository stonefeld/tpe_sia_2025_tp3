import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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


def plot_errors(errors, filename):
    plt.figure(figsize=(10, 6))

    for optimizer, error in errors.items():
        plt.plot(error, label=optimizer, linewidth=2)

    plt.xlabel("Época")
    plt.ylabel("Error")
    plt.title("Error de entrenamiento por época")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)


def main():
    X_train, Y_train, num_outputs = cargar_imagenes_y_etiquetas("assets/training_set")

    input_size = len(X_train[0])
    capas = [input_size, 30, num_outputs]

    sgd = SGD(learning_rate=0.01)
    momentum = Momentum(learning_rate=0.001, momentum=0.8)
    adam = Adam(learning_rate=0.001, layers=capas)
    optimizers = [sgd, momentum, adam]

    train_errors = {}

    for optimizer in optimizers:
        mlp = PerceptronMulticapa(capas, tita=tanh, tita_prime=tanh_prime, optimizer=optimizer)
        results = mlp.train(X_train, Y_train, epocas=1000, tolerancia=0.001)
        train_errors[optimizer.__class__.__name__] = results["errors"]

    plot_errors(train_errors, "train_errors.png")


if __name__ == "__main__":
    main()
