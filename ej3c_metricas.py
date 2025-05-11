import csv

def calcular_metricas_multiclase(path_csv):
    with open(path_csv, newline='') as csvfile:
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

    print("Matriz de Confusión:")
    print("   " + "  ".join(f"{c:2}" for c in clases))
    for r in clases:
        print(f"{r:2} " + "  ".join(f"{matriz[r][c]:2}" for c in clases))

    total = sum(sum(filas.values()) for filas in matriz.values())
    correctos = sum(matriz[c][c] for c in clases)
    accuracy = correctos / total
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # Reemplazá el nombre por el CSV generado por ej3c.py
    calcular_metricas_multiclase("resultados_test_3c.csv")
