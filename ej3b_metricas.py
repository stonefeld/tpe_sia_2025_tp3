import csv

from PIL import Image, ImageDraw, ImageFont


def calcular_metricas_binarias(path_csv):
    TP = TN = FP = FN = 0

    with open(path_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            esperado = int(row["Esperado"])
            predicho = int(row["Predicho"])

            if esperado == 1 and predicho == 1:
                TP += 1
            elif esperado == -1 and predicho == -1:
                TN += 1
            elif esperado == -1 and predicho == 1:
                FP += 1
            elif esperado == 1 and predicho == -1:
                FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    tasa_tp = TP / (TP + FN) if TP + FN > 0 else 0
    tasa_fp = FP / (FP + TN) if FP + TN > 0 else 0

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"Tasa de Verdaderos Positivos: {tasa_tp:.4f}")
    print(f"Tasa de Falsos Positivos: {tasa_fp:.4f}")

    dibujar_matriz_confusion_binaria(TN, FP, FN, TP)


def dibujar_matriz_confusion_binaria(TN, FP, FN, TP, nombre_archivo="matriz_confusion_b.png"):
    clases = [-1, 1]
    tam_celda = 80
    margen = 100
    ancho = margen + 2 * tam_celda
    alto = margen + 2 * tam_celda

    imagen = Image.new("RGB", (ancho, alto), "white")
    draw = ImageDraw.Draw(imagen)

    try:
        fuente = ImageFont.truetype("arial.ttf", 22)
    except:
        fuente = ImageFont.load_default()

    draw.text((margen + tam_celda // 2 - 40, 10), "Predicción", fill="black", font=fuente)
    draw.text((10, margen + tam_celda // 2), "Real", fill="black", font=fuente)

    draw.text((margen + 0 * tam_celda + 30, margen // 2), "-1", fill="black", font=fuente)
    draw.text((margen + 1 * tam_celda + 30, margen // 2), " 1", fill="black", font=fuente)

    draw.text((margen // 2, margen + 0 * tam_celda + 30), "-1", fill="black", font=fuente)
    draw.text((margen // 2, margen + 1 * tam_celda + 30), " 1", fill="black", font=fuente)

    valores = [[TN, FP], [FN, TP]]
    max_val = max(max(fila) for fila in valores)

    for i in range(2):
        for j in range(2):
            val = valores[i][j]
            x0 = margen + j * tam_celda
            y0 = margen + i * tam_celda
            x1 = x0 + tam_celda
            y1 = y0 + tam_celda

            intensidad = int(255 * (1 - val / max_val)) if max_val > 0 else 255
            color = (intensidad, intensidad, 255)
            draw.rectangle([x0, y0, x1, y1], fill=color, outline="black")

            texto = str(val)
            bbox = fuente.getbbox(texto)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x0 + (tam_celda - w) // 2, y0 + (tam_celda - h) // 2), texto, fill="black", font=fuente)

    imagen.save(nombre_archivo)
    print(f"\nImagen guardada como: {nombre_archivo}")


if __name__ == "__main__":
    calcular_metricas_binarias("resultados_digitos.csv")
