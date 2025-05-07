import math
import numpy as np

from src.perceptron import PerceptronMulticapa


def tanh(h):
    return math.tanh(h)


def tanh_prime(h):
    return 1 - math.tanh(h) ** 2


def main():
    xor_data = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    xor_labels = np.array([-1, 1, 1, -1])

    mlp = PerceptronMulticapa(capas=[2, 2, 1], tita=tanh, tita_prime=tanh_prime, alpha=0.1)
    mlp.train(xor_data, xor_labels, epocas=1000, tolerancia=0.005)

    print("\nResultados sobre el conjunto de entrenamiento:")
    for i, x in enumerate(xor_data):
        salida = mlp.predict(x)
        predicho = round(salida[0])
        esperado = xor_labels[i]
        print(f"Input: [{', '.join(f'{i:>2}' for i in x)}] => Predicho: {predicho:>2}, Esperado: {esperado:>2}", end="")
        if predicho == esperado:
            print(" ✅")
        else:
            print(" ❌")
        print(f"\tSalida: {salida[0]:8.5f}")


if __name__ == "__main__":
    main()
