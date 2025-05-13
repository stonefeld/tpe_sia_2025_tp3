import csv

from PIL import Image, ImageDraw, ImageFont


def calcular_matriz_confusion(path_csv):
    with open(path_csv, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        y_true = []
        y_pred = []

        for row in reader:
            y_true.append(int(row["Esperado"]))
            y_pred.append(int(row["Predicho"]))

    clases = sorted(set(y_true + y_pred))
    matriz = {i: {j: 0 for j in clases} for i in clases}

    for yt, yp in zip(y_true, y_pred):
        matriz[yt][yp] += 1

    return matriz, clases


def dibujar_matriz_confusion(matriz, clases, nombre_archivo="matriz_confusion.png"):
    tam_celda = 60
    margen = 100
    ancho = margen + len(clases) * tam_celda
    alto = margen + len(clases) * tam_celda

    imagen = Image.new("RGB", (ancho, alto), "white")
    draw = ImageDraw.Draw(imagen)

    try:
        fuente = ImageFont.truetype("arial.ttf", 20)
    except:
        fuente = ImageFont.load_default()

    draw.text((margen + (len(clases) * tam_celda) // 2 - 40, 10), "Predicción", fill="black", font=fuente)
    draw.text((10, margen + (len(clases) * tam_celda) // 2), "Real", fill="black", font=fuente)

    for idx, clase in enumerate(clases):
        x = margen + idx * tam_celda + tam_celda // 2
        y = margen // 2
        draw.text((x - 10, y), str(clase), fill="black", font=fuente)

        x = margen // 2
        y = margen + idx * tam_celda + tam_celda // 2
        draw.text((x - 10, y), str(clase), fill="black", font=fuente)

    max_val = max(matriz[r][c] for r in clases for c in clases)
    for i, r in enumerate(clases):
        for j, c in enumerate(clases):
            val = matriz[r][c]
            x0 = margen + j * tam_celda
            y0 = margen + i * tam_celda
            x1 = x0 + tam_celda
            y1 = y0 + tam_celda

            intensidad = int(255 * (1 - val / max_val))  # más oscuro = más grande
            color = (intensidad, intensidad, 255)
            draw.rectangle([x0, y0, x1, y1], fill=color, outline="black")

            texto = str(val)
            bbox = fuente.getbbox(texto)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x0 + (tam_celda - w) // 2, y0 + (tam_celda - h) // 2), texto, fill="black", font=fuente)

    imagen.save(nombre_archivo)
    print(f"Imagen guardada como: {nombre_archivo}")


if __name__ == "__main__":
    archivo = "resultados_test_3c.csv"
    matriz, clases = calcular_matriz_confusion(archivo)
    dibujar_matriz_confusion(matriz, clases)
