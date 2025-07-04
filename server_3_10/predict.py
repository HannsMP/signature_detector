import numpy as np


def percentage_np(data):
    argmax = np.argmax(data)
    d = data[argmax] * 0.5
    print(argmax, data.shape, d)

    if argmax == 0:
        return (0.5 + d) * 100

    if argmax == 1:
        return (0.5 - d) * 100


if __name__ == "__main__":
    a = np.array([[0.13719724, 0.8628027]])
    p = percentage_np(a[0])
    print(p)
