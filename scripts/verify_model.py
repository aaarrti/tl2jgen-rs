import ctypes
import numpy as np


def main():

    X = np.load("data/X_test.npy")
    y = np.load("data/y_pred.npy")
    idx = np.random.choice(np.arange(0, X.shape[0]), 100)

    X = X[idx]
    y = y[idx]

    mylib = ctypes.CDLL("./data/libmodel.so")

    mylib.predict.argtypes = [ctypes.POINTER(ctypes.c_double)]
    mylib.predict.restype = ctypes.c_double

    for X_i, y_i in zip(X, y, strict=True):

        data = np.asarray(X_i, dtype=np.float64)
        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        result = mylib.predict(data_ptr)
        np.testing.assert_allclose(result, y_i)


if __name__ == "__main__":
    main()
