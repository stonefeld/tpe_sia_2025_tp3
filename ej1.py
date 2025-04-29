import json

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from src.perceptron import PerceptronSimple


def tita(x):
    return 1 if x >= 0 else -1


def prepare_data(raw_x):
    return [[1] + list(row) for row in raw_x]


def plot_timelapse(timelapse: dict, filename: str):
    data = timelapse["data"]
    labels = timelapse["labels"]
    lapse = timelapse["lapse"]
    epochs = sorted(int(k) for k in lapse.keys())

    fig, ax = plt.subplots()

    def animate(epoch_idx):
        ax.clear()
        epoch = epochs[epoch_idx]
        weights = lapse[epoch]["weights"]

        # Plot points
        for xi, yi in zip(data, labels):
            ax.plot(xi[1], xi[2], "rx" if yi == 1 else "bo")  # Red for +1

        x_vals = [-2, 2]
        if weights[2] != 0:
            y_vals = [-(weights[1] * x + weights[0]) / weights[2] for x in x_vals]
            ax.plot(x_vals, y_vals, "k-")  # Black line

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title(f"Epoch {epoch}")
        ax.grid(True)

    ani = animation.FuncAnimation(fig, animate, frames=len(epochs), interval=500)
    ani.save(filename, writer="ffmpeg", fps=2)

    plt.show()


def main():
    # AND problem
    x_and = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
    y_and = [-1, -1, -1, 1]

    data_and = prepare_data(x_and)
    pand = PerceptronSimple(input_size=2, tita=tita)

    print("\n============== AND ==============")
    timelapse_and = pand.train(data_and, y_and)
    print(f"Entrenamiento completado en la época {len(timelapse_and['lapse'])}")

    print("Predicciones finales:")
    for xi, yi in zip(data_and, y_and):
        y_hat = pand.predict(xi)
        print(f"- Input: ({xi[1]:>2}, {xi[2]:>2}) -> Predicción: {y_hat:>2}, Esperado: {yi:>2}")

    with open("timelapse_and.json", "w") as f:
        json.dump(timelapse_and, f, indent=2)

    plot_timelapse(timelapse_and, "timelapse_and.mp4")

    # ==========================
    # XOR problem
    x_xor = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
    y_xor = [1, 1, -1, -1]

    data_xor = prepare_data(x_xor)
    pxor = PerceptronSimple(input_size=2, tita=tita)
    print("\n============== XOR ==============")
    timelapse_xor = pxor.train(data_xor, y_xor)
    print(f"Entrenamiento completado en la época {len(timelapse_xor['lapse'])}")

    print("Predicciones finales:")
    for xi, yi in zip(data_xor, y_xor):
        y_hat = pxor.predict(xi)
        print(f"- Input: ({xi[1]:>2}, {xi[2]:>2}) -> Predicción: {y_hat:>2}, Esperado: {yi:>2}")

    with open("timelapse_xor.json", "w") as f:
        json.dump(timelapse_xor, f, indent=2)

    plot_timelapse(timelapse_xor, "timelapse_xor.mp4")


if __name__ == "__main__":
    main()
