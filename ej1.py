import csv
import random as rnd

import matplotlib.pyplot as plt


# define the tita function that gets the weighted sum
# and returns the activation for an AND gate
def tita(x):
    return 1 if x >= 0 else -1


def plot_decision_boundary(w, x, epoch):
    plt.clf()

    # Separate points by class
    for point in x:
        if point[-1] == 1:
            plt.plot(point[1], point[2], "ro")  # Red for 1
        else:
            plt.plot(point[1], point[2], "bo")  # Blue for -1

    # Plot the decision boundary
    # w0 + w1*x1 + w2*x2 = 0 -> x2 = -(w1/w2)*x1 - (w0/w2)
    x1_vals = [-2, 2]  # Range for x-axis
    if w[2] != 0:
        x2_vals = [-(w[1] * x + w[0]) / w[2] for x in x1_vals]
        plt.plot(x1_vals, x2_vals, "k-")  # Black line

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title(f"Epoch {epoch}")
    plt.grid(True)
    plt.pause(0.5)  # Pause to animate


def main():
    # read the 'and.csv' file
    with open("and.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        x = list(reader)

    # add an extra x0 value to shift the function (bias term)
    x = [[1] + list(map(int, row)) for row in x]

    m = len(x)
    cols = len(x[0])

    # initialize weights w to small random values
    w = [rnd.uniform(0.1, 1) for _ in range(cols)]

    # set learning rate
    n = 0.05

    # set number of epochs
    epochs = 100

    plt.ion()  # Turn interactive mode on for live plotting

    for epoch in range(epochs):
        total_error = 0
        for u in range(m):
            # calculate the weighted sum
            h = sum(w[i] * x[u][i] for i in range(cols - 1))  # Only features, no label

            # compute activation given by tita
            y_hat = tita(h)

            # calculate difference
            delta = x[u][-1] - y_hat

            # update the weights
            for i in range(cols - 1):
                w[i] += n * delta * x[u][i]

            total_error += abs(delta)

        plot_decision_boundary(w, x, epoch)

        if total_error == 0:
            print(f"Training complete at epoch {epoch}")
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
