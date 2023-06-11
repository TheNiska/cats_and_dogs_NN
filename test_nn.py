import numpy as np
import pandas as pd
import math
from PIL import Image


def visualize_parameters(w1, w2):
    path = "visualized_parameters/"

    def visualize_w2(w1, w2):
        # 1 x 4
        mn = np.amin(w2)
        w2 = w2 - mn

        mx = np.amax(w2)  # 0 .. mx
        w2 = w2 * (1 / mx)  # 0 .. 1

        arr = w2.T * w1
        arr = np.sum(arr, axis=0, keepdims=True)
        print(arr.shape)

        arr_min = np.amin(arr)
        arr = arr - arr_min
        arr_max = np.amax(arr)
        arr = arr * (255 / arr_max)

        print(np.amin(arr), np.amax(arr))

        arr = arr.reshape(28, 28)
        arr = np.array(arr, dtype='uint8')
        img = Image.fromarray(arr)
        img = img.resize((196, 196))
        img.save(f"{path}w2.png")

    # 4 x 784
    mn = np.amin(w1)
    w1 = w1 - mn
    mx = np.amax(w1)  # 0 .. mx
    w1 = w1 * (255 / mx)  # 0 .. 255

    for i in range(w1.shape[0]):
        arr = w1[i, :].reshape(28, 28)
        arr = np.array(arr, dtype='uint8')
        img = Image.fromarray(arr)
        img = img.resize((196, 196))
        img.save(f"{path}{i}.png")

    visualize_w2(w1, w2)


def main_test():
    np.set_printoptions(precision=3)
    with np.load('params.npz') as data:
        w1 = data['w1']
        b1 = data['b1']
        w2 = data['w2']
        b2 = data['b2']
        w3 = data['w3']
        b3 = data['b3']
    print(w2.shape)
    visualize_parameters(w1, w2, number)
    with np.load('mnist_data.npz') as data:
        X = data['X']
        Y = data['Y']

    n0, m = X.shape
    print(f"Number of training examples: {m}")

    t = 0

    BATH = 1000
    assert m % BATH == 0

    k = 0
    difference = 0

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
        difference += 1 - (sum_trues / sum_nums)
        k += 1
        #print(f"{sum_trues} out of {sum_nums}")

        # print("%0.3f" % (np.sum(isSame) / BATH))
        # ----------------------------------------------------------------
        t += BATH

    print(f"Погрешность: {difference / k}")


if __name__ == '__main__':
    main_test()
