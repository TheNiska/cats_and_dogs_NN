from PIL import Image
import numpy as np
import PIL.ImageOps as imOp

np.set_printoptions(precision=3)
path = 'custom images/'
fname = 'one.png'
name, ext = fname.split('.')
img = Image.open(path + fname).convert('L').resize((28, 28))
img = imOp.invert(img)
img.save(path + name + "_grey." + ext)

X = np.array(img.getdata()).reshape(784, 1)
X = (X - 127.5) / 127.5

with np.load('params.npz') as data:
    w1 = data['w1']
    b1 = data['b1']
    w2 = data['w2']
    b2 = data['b2']
    w3 = data['w3']
    b3 = data['b3']

z1 = np.dot(w1, X) + b1
a1 = np.tanh(z1)

z2 = np.dot(w2, a1) + b2
a2 = np.tanh(z2)

z3 = np.dot(w3, a2) + b3
a3 = 1 / (1 + np.exp(-z3))

print(f"Probability of 1: {a3}")

