import matplotlib.pyplot as plt
import csv

epochs = []
train = []
test = []

with open("accuracy_vs_epoch.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        train.append(float(row["train_accuracy"]))
        test.append(float(row["test_accuracy"]) if row["test_accuracy"] else None)

plt.plot(epochs, train, label="train")
plt.plot(epochs, test, label="test")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("model accuracy")
plt.legend()
plt.grid()
plt.show()
