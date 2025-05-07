import math

from src.perceptron import PerceptronMulticapa


def tanh(h):
    return math.tanh(h)


def tanh_prime(h):
    return 1 - math.tanh(h) ** 2


def main():
    xor_data = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    xor_labels = [-1, 1, 1, -1]

    mlp = PerceptronMulticapa(capas=[2, 2, 1], tita=tanh, tita_prime=tanh_prime, alpha=0.1)
    mlp.train(xor_data, xor_labels)

    for x in xor_data:
        print(f"{x} => {round(mlp.predict(x)[0])}")


if __name__ == "__main__":
    main()
