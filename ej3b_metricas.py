import csv

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
    Tasa_TP = TP / (TP + FN) if TP + FN > 0 else 0
    Tasa_FP = FP / (FP + TN) if FP + TN > 0 else 0

    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"Tasa de Verdaderos Positivos: {Tasa_TP:.4f}")
    print(f"Tasa de Falsos Positivos: {Tasa_FP:.4f}")

if __name__ == "__main__":
    calcular_metricas_binarias("resultados_digitos.csv")
