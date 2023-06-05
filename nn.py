import numpy as np


'''
this shitty NN doesn't work because it's structure isn't suted for that
kind of task. I will add more layers.
'''


def main():
    # load data from file
    X_dict = np.load('data_X.npz')
    data = X_dict['X']
    np.random.shuffle(data.T)

    data_X = data[:-1, :]
    data_Y = data[-1, :].reshape(1, 30000)
    print(data_X.shape)
    print(data_Y.shape)

    return 0

    X = np.zeros(shape=(22500, 30000))
    Y = np.zeros(shape=(1, 30000))

    X = (data_X - 127.5) / 127.5
    Y = data_Y

    w_shape_0 = X.shape[0]
    w_shape_1 = 1

    X_train = X[:, :29000]
    X_test = X[:, 29000:]

    Y_train = Y[:, :29000]
    Y_test = Y[:, 29000:]

    m = 29000

    W = np.random.randn(w_shape_0, w_shape_1) * 0.1

    b = 0
    epsilon = 0.0000001
    ALPHA = 0.01
    for i in range(5):
        Z = np.dot(W.T, X_train) + b
        assert Z.shape == (1, 29000)
        print(Z)

        A = 1 / (1 + math.e**(-Z))
        assert Z.shape == (1, 29000)
        assert Y_train.shape == Z.shape
        print(A)

        j_cost = (Y_train * np.log(A + epsilon) +
                  (1 - Y_train) * (np.log(1 - A + epsilon)))
        j_cost = np.sum(j_cost) / -m
        print(j_cost)

        dz = A - Y_train
        dw = (1 / m) * np.dot(X_train, dz.T)
        db = (1 / m) * np.sum(dz)
        W = W - ALPHA * dw
        b = b - ALPHA * db

    Z = np.dot(W.T, X_test) + b
    A = 1/(1 + math.e**(-Z))

    difference = np.sum(np.absolute(Y_test - A))
    print(difference)

    np.save('weights', W)
    print(b)


if __name__ == '__main__':
    main()
