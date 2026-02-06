import numpy as np


def cholesky(a):
    from gpaw.gpu import cupy as cp
    return cp.ndarray(np.linalg.cholesky(a._data))


def inv(a):
    from gpaw.gpu import cupy as cp
    return cp.ndarray(np.linalg.inv(a._data))


def eigh(a, UPLO):
    from gpaw.gpu import cupy as cp
    eigvals, eigvecs = np.linalg.eigh(a._data, UPLO)
    return cp.ndarray(eigvals), cp.ndarray(eigvecs.T.copy().T)
