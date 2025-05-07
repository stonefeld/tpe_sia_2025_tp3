import math
import os
import re

from PIL import Image

from src.perceptron import PerceptronMulticapa


def tanh(h):
    return math.tanh(h)


def tanh_prime(salida):
    return 1 - salida**2


def cargar_imagen_como_vector(path_imagen):
    with Image.open(path_imagen) as img:
        img = img.convert("L")  # escala de grises
        # img = img.resize((5, 7))  # redimensionamos a 5x7 si no está ya

        pixeles = list(img.getdata())
        # Convertimos a binario: 0 para oscuro, 1 para claro (o invertí si querés)
        binarizado = [1 if p < 128 else 0 for p in pixeles]

        return binarizado


def cargar_imagenes_y_etiquetas(carpeta):
    imagenes = []
    etiquetas = []

    archivos = [f for f in os.listdir(carpeta) if f.endswith(".png")]
    archivos.sort()  # para que el orden sea estable

    patron = re.compile(r"imagen_(\d+)_\w+\.png")  # coincide con imagen_3_1.png, imagen_7_b.png, etc.

    # Find the range of numbers to determine output size
    min_numero = float('inf')
    max_numero = float('-inf')
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
        etiqueta = [-1] * num_outputs
        etiqueta[numero - min_numero] = 1  # Adjust index based on min_numero

        imagenes.append(vector)
        etiquetas.append(etiqueta)

    return imagenes, etiquetas, num_outputs


def main():
    data, labels, num_outputs = cargar_imagenes_y_etiquetas("assets/numeros")

    input_size = len(data[0])
    mlp = PerceptronMulticapa([input_size, 15, num_outputs], alpha=0.1, tita=tanh, tita_prime=tanh_prime)

    print("Entrenando con múltiples imágenes por dígito...")
    mlp.train(data, labels, epocas=1000, tolerancia=0.005)

    # ahora cargamos de `assets/numeros_test` el archivo `imagen_8.png` y vemos qué predice
    test_data, test_labels, _ = cargar_imagenes_y_etiquetas("assets/numeros_tests")
    print("\nResultados sobre el conjunto de test:")
    for i, x in enumerate(test_data):
        salida = mlp.forward(x)[-1]
        pred = salida.index(max(salida))
        esperado = test_labels[i].index(1)
        print(f"Imagen {i}: Esperado: {esperado}, Predicho: {pred}", end="")
        if pred == esperado:
            print(" ✅")
        else:
            print(" ❌")
        print(f"\tSalida: [{', '.join(f'{s:8.5f}' for s in salida)}]")


if __name__ == "__main__":
    main()
