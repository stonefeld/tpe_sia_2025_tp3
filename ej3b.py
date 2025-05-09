import math
import numpy as np
import csv

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

    etiquetas = np.array([1 if i % 2 == 1 else -1 for i in range(10)])
    return np.array(digitos), etiquetas


def main():
    data, labels = cargar_digitos_y_etiquetas()

    # Red de 35 entradas, 10 neuronas ocultas, 1 salida
    mlp = PerceptronMulticapa(capas=[35, 10, 1], tita=tanh, tita_prime=tanh_prime, alpha=0.1)
    mlp.train(data, labels, epocas=1000, tolerancia=0.005)

    # Guardar resultados en CSV
    with open('resultados_digitos.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # Escribir encabezados
        writer.writerow(['digito', 'predicho', 'esperado', 'salida_exacta', 'correcto'])
        
        # Imprimir y guardar resultados
        print("\nResultados del entrenamiento:")
        for i, x in enumerate(data):
            salida = mlp.predict(x)
            predicho_valor = round(salida[0])
            predicho = "IMPAR" if predicho_valor > 0 else "PAR"
            esperado_valor = labels[i]
            esperado = "IMPAR" if esperado_valor > 0 else "PAR"
            correcto = predicho == esperado
            
            # Guardar en CSV
            writer.writerow([i, predicho, esperado, f"{salida[0]:.5f}", correcto])
            
            # Imprimir en consola
            resultado = f"Dígito {i}: Predicho: {predicho}, Esperado: {esperado}"
            if correcto:
                resultado += " ✅"
            else:
                resultado += " ❌"
            resultado += f"\n\tSalida: {salida[0]:8.5f}\n"
            print(resultado, end="")


if __name__ == "__main__":
    main()
