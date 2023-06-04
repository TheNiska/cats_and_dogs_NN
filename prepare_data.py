from PIL import Image
import numpy as np
import os

cats_pth = "Animal Images/cats/"
dogs_pth = "Animal Images/dogs/"

X = np.zeros(shape=(22501, 30000), dtype=np.uint8)
Y = np.zeros(shape=(30000,), dtype=np.uint8)

i = 0
# loop for reading cats images
for filename in os.listdir(cats_pth):
    if i >= 15000:
        break
    if i % 450 == 0:
        print(f"Done {int(i / 29999 * 100)} %")

    image = Image.open(os.path.join(cats_pth, filename))
    x = np.asarray(image, dtype=np.uint8)

    # get rid of an image if it's wrong size
    if x.shape != (150, 150):
        os.remove(os.path.join(cats_pth, filename))
        continue

    assert x.shape == (150, 150)
    x = x.reshape(150 * 150,)
    ...
    X[:-1, i] = x
    Y[i] = 1
    i += 1

# loop for reading dogs images
for filename in os.listdir(dogs_pth):
    if i >= 30000:
        break
    if i % 450 == 0:
        print(f"Done {int(i / 29999 * 100)} %")

    image = Image.open(os.path.join(dogs_pth, filename))
    x = np.asarray(image, dtype=np.uint8)

    if x.shape != (150, 150):
        os.remove(os.path.join(dogs_pth, filename))
        continue

    assert x.shape == (150, 150)
    x = x.reshape(150 * 150,)
    ...
    X[:-1, i] = x
    Y[i] = 0
    i += 1

# cats: [0 : 15000]
# dogs: [15000: 30000]

print(X)
X[-1, :] = Y
print(X)

np.savez_compressed('data_X', X=X)
print("DONE !!!")












