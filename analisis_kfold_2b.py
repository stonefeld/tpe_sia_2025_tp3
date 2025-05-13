import math
import numpy as np
import matplotlib.pyplot as plt
from ej2b import k_fold_cross_validation_nolineal, sigmoid, sigmoid_derivative

def main():
    # Cargar conjunto
    conjunto = np.loadtxt("assets/conjunto.csv", delimiter=",", skiprows=1)
    x = conjunto[:, :-1]
    y = conjunto[:, -1]

    # Normalizar salidas
    min_y = min(y)
    max_y = max(y)
    y = np.array([(i - min_y) / (max_y - min_y) for i in y])

    # Valores de K a probar
    ks = [2, 5, 8, 16, 28]
    errores_train = []
    errores_test = []

    for k in ks:
        print(f"Evaluando k = {k}")
        avg_train_error, avg_test_error = k_fold_cross_validation_nolineal(
            x,
            y,
            k=k,
            tita=sigmoid,
            tita_prime=sigmoid_derivative,
            learning_rate=5e-3,
            epochs=10000,
        )
        errores_train.append(avg_train_error)
        errores_test.append(avg_test_error)

    # Graficar
    ancho = 0.35
    posiciones = np.arange(len(ks))

    plt.figure(figsize=(8, 5))
    plt.bar(posiciones, errores_train, width=ancho, label="entrenamiento")
    plt.bar(posiciones + ancho, errores_test, width=ancho, label="testeo")
    plt.xticks(posiciones + ancho / 2, ks)
    plt.xlabel("Cantidad de particiones (k)")
    plt.ylabel("Error promedio")
    plt.title("Error promedio de entrenamiento y testeo según k (Ejercicio 2b)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grafico_error_vs_k_ej2b.png")
    plt.show()

if __name__ == "__main__":
    main()
