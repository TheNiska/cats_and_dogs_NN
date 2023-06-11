import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


# function to make labels data binary
def y_to_one_number(y, num=1):
    isNum = y[:, :] == num
    new_y = (y + 1) * isNum // (num + 1)

    return new_y


# fuction for showing image of data
def show_image(arr):
    img = arr.reshape(28, 28)
    img = Image.fromarray(img.astype('uint8'), 'L').resize((350, 350))
    img.save('img0.jpg')


def make_X_and_Y(number):
    # reading and changing the shape of the data
    df = pd.read_csv('train.csv')
    data = df.to_numpy().T

    # making correct X and Y for training
    X = data[1:, :]
    Y = data[0, :].reshape(1, 42000)

    # normalizing X
    X = (X - 127.5) / 127.5

    # binary Y only for one number
    Y = y_to_one_number(Y, num=number)

    np.savez('mnist_data', X=X, Y=Y)


if __name__ == '__main__':
    make_X_and_Y(0)
