import copy
import math
import random

from src.perceptron import PerceptronMulticapa


def tanh(h):
    return math.tanh(h)


def tanh_prime(h):
    return 1 - math.tanh(h) ** 2


def cargar_digitos_y_etiquetas(path="assets/digitos.txt"):
    with open(path) as f:
        lines = [list(map(int, line.strip().split())) for line in f if line.strip()]

    # Cada dígito tiene 7 filas, 10 dígitos en total
    digitos = []
    for i in range(0, len(lines), 7):
        right = i + 7
        bloque = lines[i:right]  # 7 filas
        flatten = [bit for fila in bloque for bit in fila]
        digitos.append(flatten)

    etiquetas = []
    for i in range(10):
        etiquetas.append(1 if i % 2 == 1 else -1)

    return digitos, etiquetas


def etiquetas_one_hot():
    etiquetas = []
    for i in range(10):
        vector = [-1] * 10
        vector[i] = 1
        etiquetas.append(vector)
    return etiquetas


def agregar_ruido(patron, probabilidad=0.1):
    ruidoso = copy.deepcopy(patron)
    for i in range(len(ruidoso)):
        if random.random() < probabilidad:
            ruidoso[i] = 1 - ruidoso[i]  # invierte el bit
    return ruidoso


def main():
    data, labels = cargar_digitos_y_etiquetas()
    labels_onehot = etiquetas_one_hot()

    # Red: 35 entradas, 20 ocultas, 10 salida
    mlp = PerceptronMulticapa([35, 20, 10], alpha=0.1, tita=tanh, tita_prime=tanh_prime)

    # Entrenamiento
    mlp.train(data, labels_onehot)

    for i, x in enumerate(data):
        salida = mlp.forward(x)[-1]
        pred = salida.index(max(salida))
        print(f"Esperado: {i}, Predicho: {pred}")

    print("Con ruido:")
    for i, x in enumerate(data):
        x_ruidoso = agregar_ruido(x, 0.1)
        salida = mlp.forward(x_ruidoso)[-1]
        pred = salida.index(max(salida))
        print(f"Esperado: {i}, Predicho: {pred}")


if __name__ == "__main__":
    main()
