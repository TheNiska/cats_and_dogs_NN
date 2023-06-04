import numpy as np

a = np.array([[56.0, 0.0,   4.4,  68.0],
              [1.2,  104.0, 52.0, 8.0],
              [1.8,  135.0, 99.0, 0.9]])


rand_arr = np.random.randn(5, 1)
assert rand_arr.shape == (5, 1)
print(rand_arr)
b = np.array([[1],
              [2],
              [3],
              [4]])
print(b.shape)
print(b)
print(b+10)


c = np.array([1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10])
print(c)
print(c.shape)
print(c.reshape(11, 1))


# vertically   axis=0
# horizontally axis=1
ver_sum = a.sum(axis=0)
print(ver_sum)
print(ver_sum.shape)

percentage = 100 * a / ver_sum.reshape(1, 4)
print(percentage)