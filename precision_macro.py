import csv
import matplotlib.pyplot as plt

epochs = []
train_precision = []
test_precision = []

with open("accuracy_precision_vs_epoch.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        train_precision.append(float(row["train_precision"]))
        test_precision.append(float(row["test_precision"]))

plt.plot(epochs, train_precision, label="Train Precision Macro")
plt.plot(epochs, test_precision, label="Test Precision Macro")
plt.xlabel("Época")
plt.ylabel("Precisión Macro")
plt.title("Precisión Macro por Época")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
