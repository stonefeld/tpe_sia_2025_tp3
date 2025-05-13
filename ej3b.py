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
        right = i + 7
        bloque = lines[i:right]  # 7 filas
        flatten = [bit for fila in bloque for bit in fila]
        digitos.append(flatten)

    etiquetas = np.array([1 if i % 2 == 1 else -1 for i in range(10)])
    return np.array(digitos), etiquetas


def main():
    data, labels = cargar_digitos_y_etiquetas()

    mlp = PerceptronMulticapa(capas=[data.shape[1], 10, 1], tita=tanh, tita_prime=tanh_prime)
    mlp.train(data, labels, epocas=1000, tolerancia=0.005)

    for i, x in enumerate(data):
        salida = mlp.predict(x)
        predicho = round(salida[0])
        paridad = "IMPAR" if predicho > 0 else "PAR"
        esperado = labels[i]
        print(f"Dígito {i}: Predicho: {predicho:>2}, Esperado: {esperado:>2}, Paridad: {paridad}", end="")
        if predicho == esperado:
            print(" ✅")
        else:
            print(" ❌")
        print(f"\tSalida: {salida[0]:8.5f}")


if __name__ == "__main__":
    main()
