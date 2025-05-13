import csv
import math
import os
import re

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


def main():
    X_train, Y_train, num_outputs = cargar_imagenes_y_etiquetas("assets/training_set")
    X_test, Y_test, _ = cargar_imagenes_y_etiquetas("assets/testing_set")

    input_size = len(X_train[0])
    capas = [input_size, 30, num_outputs]

    sgd = SGD(learning_rate=0.01)
    momentum = Momentum(learning_rate=0.001, momentum=0.8)
    adam = Adam(learning_rate=0.001, layers=capas)

    mlp = PerceptronMulticapa(capas, tita=tanh, tita_prime=tanh_prime, optimizer=momentum)

    epocas = 300
    train_accs, test_accs = [], []
    train_precs, test_precs = [], []

    for epoch in range(epocas):
        mlp.train(X_train, Y_train, epocas=1, tolerancia=0.001)

        pred_train = [mlp.forward(x)[-1] for x in X_train]
        acc_train, prec_train = calcular_metricas(pred_train, Y_train)
        train_accs.append(acc_train)
        train_precs.append(prec_train)

        pred_test = [mlp.forward(x)[-1] for x in X_test]
        acc_test, prec_test = calcular_metricas(pred_test, Y_test)
        test_accs.append(acc_test)
        test_precs.append(prec_test)

        print(f"Epoch {epoch+1:3} | Train Acc: {acc_train:.4f} Prec: {prec_train:.4f} | Test Acc: {acc_test:.4f} Prec: {prec_test:.4f}")

    with open("accuracy_precision_vs_epoch.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_accuracy", "train_precision", "test_accuracy", "test_precision"])
        for i in range(epocas):
            writer.writerow([i + 1, train_accs[i], train_precs[i], test_accs[i], test_precs[i]])

    with open("resultados_test_3c.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Imagen", "Esperado", "Predicho", "Salidas"])
        for i, (x, t) in enumerate(zip(X_test, Y_test)):
            salida = mlp.forward(x)[-1]
            predicho = np.argmax(salida)
            esperado = np.argmax(t)
            writer.writerow([i, esperado, predicho, ",".join(f"{s:.5f}" for s in salida)])


if __name__ == "__main__":
    main()
