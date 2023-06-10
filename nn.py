import numpy as np
import math


def main():
    with np.load('mnist_data.npz') as data:
        X = data['X']
        Y = data['Y']

    n0, m = X.shape
    print(f"Number of training examples: {m}")

    n1 = 72
    n2 = 32
    n3 = 1

    w1 = np.random.randn(n1, n0) * math.sqrt(1/n0)
    b1 = np.random.randn(n1, 1)

    w2 = np.random.randn(n2, n1) * math.sqrt(1/n1)
    b2 = np.random.randn(n2, 1)

    w3 = np.random.randn(n3, n2) * math.sqrt(1/n2)
    b3 = np.random.randn(n3, 1)

    ITER = 50
    ALPHA = 0.01
    BATH = 1024
    for i in range(ITER):
        t = 0
        overall_cost = 0
        print(f"-----------Epoch {i}--------------")

        while t < X.shape[1]:
            # Forward prop ---------------------------------------------------
            z1 = np.dot(w1, X[:, t:t+BATH]) + b1
            a1 = np.tanh(z1)

            z2 = np.dot(w2, a1) + b2
            a2 = np.tanh(z2)

            z3 = np.dot(w3, a2) + b3
            a3 = 1 / (1 + np.exp(-z3))
            # ----------------------------------------------------------------

            cost = ((-1/BATH) * (np.dot(Y[:, t:t+BATH], np.log(a3).T) +
                                 np.dot((1 - Y[:, t:t+BATH]), np.log(1-a3).T)))
            overall_cost += cost

            # Back prop ------------------------------------------------------
            dz3 = a3 - Y[:, t:t+BATH]
            dw3 = (1 / BATH) * np.dot(dz3, a2.T)
            db3 = (1 / BATH) * np.sum(dz3, axis=1, keepdims=True)

            dz2 = np.dot(w3.T, dz3) * (1 - np.power(a2, 2))
            dw2 = (1 / BATH) * (np.dot(dz2, a1.T))
            db2 = np.sum(dz2, axis=1, keepdims=True) * (1 / BATH)

            dz1 = np.dot(w2.T, dz2) * (1 - np.power(a1, 2))
            dw1 = (1 / BATH) * np.dot(dz1, X[:, t:t+BATH].T)
            db1 = (1 / BATH) * np.sum(dz1, axis=1, keepdims=True)

            w1 = w1 - ALPHA * dw1
            b1 = b1 - ALPHA * db1
            w2 = w2 - ALPHA * dw2
            b2 = b2 - ALPHA * db2
            w3 = w3 - ALPHA * dw3
            b3 = b3 - ALPHA * db3
            # --------------------------------------------------------------------

            t += BATH
        print(f"Overall cost: {overall_cost}")

    # saving parameters ------------------------------------
    np.savez('params', w1=w1, w2=w2, w3=w3, b1=b1, b2=b2, b3=b3)

if __name__ == '__main__':
    main()
