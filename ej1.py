import json

from src.perceptron import Perceptron


def tita(x):
    return 1 if x >= 0 else -1


def prepare_data(raw_x):
    return [[1] + list(row) for row in raw_x]


def main():
    # AND problem
    x_and = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
    y_and = [-1, -1, -1, 1]

    data_and = prepare_data(x_and)
    pand = Perceptron(input_size=2, tita=tita)

    print("\n============== AND ==============")
    timelapse = pand.train(data_and, y_and)
    print(f"Entrenamiento completado en la época {len(timelapse)}")

    print("Predicciones finales:")
    for xi, yi in zip(data_and, y_and):
        y_hat = pand.predict(xi)
        print(f"- Input: ({xi[1]:>2}, {xi[2]:>2}) -> Predicción: {y_hat:>2}, Esperado: {yi:>2}")

    with open("timelapse_and.json", "w") as f:
        json.dump(timelapse, f, indent=2)

    # ==========================
    # XOR problem
    x_xor = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
    y_xor = [1, 1, -1, -1]

    data_xor = prepare_data(x_xor)
    pxor = Perceptron(input_size=2, tita=tita)
    print("\n============== XOR ==============")
    timelapse = pxor.train(data_xor, y_xor)
    print(f"Entrenamiento completado en la época {len(timelapse)}")

    print("Predicciones finales:")
    for xi, yi in zip(data_xor, y_xor):
        y_hat = pxor.predict(xi)
        print(f"- Input: ({xi[1]:>2}, {xi[2]:>2}) -> Predicción: {y_hat:>2}, Esperado: {yi:>2}")

    with open("timelapse_xor.json", "w") as f:
        json.dump(timelapse, f, indent=2)


if __name__ == "__main__":
    main()
