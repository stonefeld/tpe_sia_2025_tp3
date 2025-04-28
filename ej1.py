import csv
import random as rnd

import matplotlib.pyplot as plt


# define the tita function that gets the weighted sum
# and returns the activation for an AND gate
def tita(x):
    if x >= 0:
        return 1
    else:
        return -1


def main():
    # read the 'and.csv' file
    with open("and.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        x = list(reader)

    # add and extra x0 value to shift the function
    x = [[1] + list(map(int, row)) for row in x]

    # the number of training examples
    m = len(x)
    cols = len(x[0]) - 1

    # initialize weights w to small random values
    w = [rnd.uniform(0.1, 1) for _ in range(cols)]

    # set learning rate n
    n = 0.05

    # set number of epochs
    epochs = 1000

    # set the algorithm results
    results = {}

    for epoch in range(epochs):
        for u in range(m):
            # calculate the weighted sum
            h = sum(w[i] * int(x[u][i]) for i in range(cols))

            # compute activation given by tita
            results[h] = tita(h)

            # update the weights and bias
            for i in range(cols):
                w[i] += n * (int(x[u][-1]) - results[h]) * int(x[u][i])

        # calculate perceptron error
        error = sum((int(x[u][cols]) - results[h]) for u in range(m))
        if error == 0:
            print(f"Epoch {epoch}: Weights: {w}, Error: {error}")
            break


if __name__ == "__main__":
    main()
