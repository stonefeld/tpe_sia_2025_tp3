import math
import os
import re
import csv
from datetime import datetime

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


def main():
    # TRAINING
    data, labels, num_outputs = cargar_imagenes_y_etiquetas("assets/training_set")

    input_size = len(data[0])
    layers = [input_size, 30, num_outputs]

    sgd = SGD(learning_rate=0.01)
    momentum = Momentum(learning_rate=0.001, momentum=0.8)
    adam = Adam(learning_rate=0.001, layers=layers)
    mlp = PerceptronMulticapa(layers, tita=tanh, tita_prime=tanh_prime, optimizer=momentum)

    print("Entrenando con múltiples imágenes por dígito...")
    mlp.train(data, labels, epocas=1000, tolerancia=0.001)

    # TESTING
    test_data, test_labels, _ = cargar_imagenes_y_etiquetas("assets/testing_set")
    correctos = 0

    # Crear archivo CSV con timestamp
    csv_filename = f"resultados_test_3c.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Escribir encabezados
        csv_writer.writerow(['Imagen', 'Esperado', 'Predicho', 'Salidas'])
        
        print("\nResultados sobre el conjunto de test:")
        for i, x in enumerate(test_data):
            salida = mlp.forward(x)[-1]
            predicho = np.argmax(salida)
            esperado = np.argmax(test_labels[i])
            es_correcto = predicho == esperado
            if es_correcto:
                correctos += 1
                print(f"Imagen {i}: Esperado: {esperado}, Predicho: {predicho} ✅")
            else:
                print(f"Imagen {i}: Esperado: {esperado}, Predicho: {predicho} ❌")
            print(f"\tSalida: [{', '.join(f'{s:8.5f}' for s in salida)}]")
            
            # Guardar en CSV
            csv_writer.writerow([
                i,
                esperado,
                predicho,
                ','.join(f'{s:.5f}' for s in salida)
            ])

    print(f"\nTotal de imágenes correctas: {correctos}/{len(test_data)}")
    print(f"Porcentaje de aciertos: {correctos/len(test_data)*100:.2f}%")
    print(f"\nResultados guardados en: {csv_filename}")


if __name__ == "__main__":
    main()
