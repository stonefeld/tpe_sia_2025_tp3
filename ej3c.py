import math
import os
import re
import csv
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
            print(f"Archivo ignorado por formato no válido: {archivo}")
            continue

        numero = int(match.group(1))
        etiqueta = np.full(num_outputs, -1)
        etiqueta[numero - min_numero] = 1

        imagenes.append(vector)
        etiquetas.append(etiqueta)

    return np.array(imagenes), np.array(etiquetas), num_outputs


def calcular_accuracy(preds, targets):
    correctos = sum(np.argmax(p) == np.argmax(t) for p, t in zip(preds, targets))
    return correctos / len(targets)


def main():
    # Cargar datos
    X_train, Y_train, num_outputs = cargar_imagenes_y_etiquetas("assets/training_set")
    X_test, Y_test, _ = cargar_imagenes_y_etiquetas("assets/testing_set")

    input_size = len(X_train[0])
    layers = [input_size, 30, num_outputs]

    sgd = SGD(learning_rate=0.01)
    momentum = Momentum(learning_rate=0.001, momentum=0.8)
    adam = Adam(learning_rate=0.001, layers=layers)

    mlp = PerceptronMulticapa(
        layers, tita=tanh, tita_prime=tanh_prime,
        optimizer=momentum
    )

    epocas = 300
    train_accs = []
    test_accs = []

    for epoch in range(epocas):
        mlp.train(X_train, Y_train, epocas=1, tolerancia=0.001)

        pred_train = [mlp.forward(x)[-1] for x in X_train]
        acc_train = calcular_accuracy(pred_train, Y_train)
        train_accs.append(acc_train)

        pred_test = [mlp.forward(x)[-1] for x in X_test]
        acc_test = calcular_accuracy(pred_test, Y_test)
        test_accs.append(acc_test)

        print(f"Epoch {epoch+1}/{epocas} - Train Acc: {acc_train:.4f} - Test Acc: {acc_test:.4f}")

    # Guardar en CSV
    with open("accuracy_vs_epoch.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_accuracy", "test_accuracy"])
        for i in range(epocas):
            writer.writerow([i+1, train_accs[i], test_accs[i]])

    print("\n¡Entrenamiento completo! Accuracy por época guardado en 'accuracy_vs_epoch.csv'")


if __name__ == "__main__":
    main()
