import numpy as np
import pandas as pd
import math


def main():
    np.set_printoptions(precision=3)
    with np.load('params.npz') as data:
        w1 = data['w1']
        b1 = data['b1']
        w2 = data['w2']
        b2 = data['b2']
        w3 = data['w3']
        b3 = data['b3']

    with np.load('mnist_data.npz') as data:
        X = data['X']
        Y = data['Y']

    n0, m = X.shape
    print(f"Number of training examples: {m}")

    t = 0

    BATH = 1000
    assert m % BATH == 0

    print("Precisions:")
    while t < X.shape[1]:
        # Forward prop ---------------------------------------------------
        z1 = np.dot(w1, X[:, t:t+BATH]) + b1
        a1 = np.tanh(z1)

        z2 = np.dot(w2, a1) + b2
        a2 = np.tanh(z2)

        z3 = np.dot(w3, a2) + b3
        a3 = 1 / (1 + np.exp(-z3))

        probs = a3 > 0.5
        y_probs = Y[:, t:t+BATH] == 1
        sum_nums = np.sum(y_probs)

        isSame = np.logical_not(np.logical_xor(probs, y_probs))
        isTrueNum = np.logical_and(probs, y_probs)
        sum_trues = np.sum(isTrueNum)
        print(f"{sum_trues} out of {sum_nums}")

        # print("%0.3f" % (np.sum(isSame) / BATH))
        # ----------------------------------------------------------------
        t += BATH


if __name__ == '__main__':
    main()
