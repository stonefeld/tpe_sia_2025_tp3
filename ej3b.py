import math

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


def main():
    data, labels = cargar_digitos_y_etiquetas()

    # Red de 35 entradas, 10 neuronas ocultas, 1 salida
    mlp = PerceptronMulticapa(capas=[35, 10, 1], tita=tanh, tita_prime=tanh_prime, alpha=0.1)
    mlp.train(data, labels)

    for i, x in enumerate(data):
        pred = mlp.predict(x)
        print(f"Dígito {i}: {'IMPAR' if pred[0] > 0 else 'PAR'} ({pred[0]:.3f})")


if __name__ == "__main__":
    main()
