import numpy as np

# load data from file
X_dict = np.load('data_X.npz')
X = X_dict['X']

ALPHA = 0.03
print(X.shape)

w_shape_0 = X.shape[0] - 1
w_shape_1 = 1

X_train = X[:, :29000]
X_test = X[:, 29000:]

m = 29000

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

W = np.random.randn(w_shape_0, w_shape_1)
print(f"W_shape: {W.shape}")

b = 1

for i in range(1):
    Z = np.dot(W.T, X_train[:-1, :3])
    print(Z.shape)
