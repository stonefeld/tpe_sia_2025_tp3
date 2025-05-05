import math
import os
import re

from PIL import Image

from src.perceptron import PerceptronMulticapa


def tanh(h):
    return math.tanh(h)


def tanh_prime(h):
    return 1 - math.tanh(h) ** 2


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

    for archivo in archivos:
        path = os.path.join(carpeta, archivo)
        vector = cargar_imagen_como_vector(path)

        match = patron.match(archivo)
        if not match:
            print(f"Archivo ignorado por formato no válido: {archivo}")
            continue

        numero = int(match.group(1))
        etiqueta = [-1] * 10
        etiqueta[numero] = 1

        imagenes.append(vector)
        etiquetas.append(etiqueta)

    return imagenes, etiquetas


def etiquetas_one_hot():
    etiquetas = []
    for i in range(10):
        vector = [-1] * 10
        vector[i] = 1
        etiquetas.append(vector)
    return etiquetas


def main():
    data, labels = cargar_imagenes_y_etiquetas("assets/numeros")

    input_size = len(data[0])
    mlp = PerceptronMulticapa([input_size, 40, 10], alpha=0.1, tita=tanh, tita_prime=tanh_prime)

    print("Entrenando con múltiples imágenes por dígito...")
    mlp.train(data, labels)

    # ahora cargamos de `assets/numeros_test` el archivo `imagen_8.png` y vemos qué predice
    test_data, test_labels = cargar_imagenes_y_etiquetas("assets/numeros_tests")
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


if __name__ == "__main__":
    main()
