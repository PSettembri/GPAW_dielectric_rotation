from types import SimpleNamespace

import numpy as np

import gpaw.gpu.cpupy.cublas as cublas
import gpaw.gpu.cpupy.fft as fft
import gpaw.gpu.cpupy.linalg as linalg
import gpaw.gpu.cpupy.random as random

__all__ = ['linalg', 'cublas', 'fft', 'random']


def empty(*args, **kwargs):
    return ndarray(np.empty(*args, **kwargs))


def empty_like(a):
    return ndarray(np.empty_like(a._data))


def zeros(*args, **kwargs):
    return ndarray(np.zeros(*args, **kwargs))


def ones(*args, **kwargs):
    return ndarray(np.ones(*args, **kwargs))


def asnumpy(a, out=None):
    if out is None:
        return a._data.copy()
    out[:] = a._data
    return out


def asarray(a):
    if isinstance(a, ndarray):
        return a
    return ndarray(np.array(a))


def array(a, dtype=None):
    return ndarray(np.array(a, dtype))


def dot(a, b):
    return ndarray(np.dot(a._data, b._data))


def multiply(a, b, c):
    np.multiply(a._data, b._data, c._data)


def negative(a, b):
    np.negative(a._data, b._data)


def einsum(indices, *args):
    return ndarray(
        np.einsum(
            indices,
            *(arg._data for arg in args)))


def diag(a):
    return ndarray(np.diag(a._data))


def abs(a):
    return ndarray(np.abs(a._data))


def exp(a):
    return ndarray(np.exp(a._data))


def eye(n):
    return ndarray(np.eye(n))


def triu_indices(n, k=0, m=None):
    i, j = np.triu_indices(n, k, m)
    return ndarray(i), ndarray(j)


def tri(n, k=0, dtype=float):
    return ndarray(np.tri(n, k=k, dtype=dtype))


def moveaxis(a, source, destination):
    return ndarray(np.moveaxis(a._data, source, destination))


def vdot(a, b):
    return np.vdot(a._data, b._data)


def fuse():
    return lambda func: func


class ndarray:
    def __init__(self, data):
        if isinstance(data, (float, complex, int, np.int32, np.int64,
                             np.bool_)):
            data = np.asarray(data)
        assert isinstance(data, np.ndarray), type(data)
        self._data = data
        self.dtype = data.dtype
        self.size = data.size
        self.flags = data.flags
        self.ndim = data.ndim
        self.nbytes = data.nbytes
        self.data = SimpleNamespace(ptr=data.ctypes.data)

    @property
    def shape(self):
        return self._data.shape

    @property
    def T(self):
        return ndarray(self._data.T)

    @property
    def real(self):
        return ndarray(self._data.real)

    @property
    def imag(self):
        return ndarray(self._data.imag)

    def set(self, a):
        if self.ndim == 0:
            self._data.fill(a)
        else:
            self._data[:] = a

    def get(self):
        return self._data.copy()

    def copy(self):
        return ndarray(self._data.copy())

    def all(self):
        return ndarray(self._data.all())

    def sum(self, out=None, **kwargs):
        if out is not None:
            out = out._data
        return ndarray(self._data.sum(out=out, **kwargs))

    def __repr__(self):
        return 'cp.' + np.array_repr(self._data)

    def __len__(self):
        return len(self._data)

    def __bool__(self):
        return bool(self._data)

    def __float__(self):
        return self._data.__float__()

    def __iter__(self):
        for data in self._data:
            if data.ndim == 0:
                yield ndarray(data.item())
            else:
                yield ndarray(data)

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            def convert(a):
                return a._data if isinstance(a, ndarray) else a
            index = tuple([convert(a) for a in index])
        if isinstance(index, ndarray):
            index = index._data
        if isinstance(value, ndarray):
            self._data[index] = value._data
        else:
            assert isinstance(value, (float, int, complex))
            self._data[index] = value

    def __getitem__(self, index):
        if isinstance(index, tuple):
            def convert(a):
                return a._data if isinstance(a, ndarray) else a
            index = tuple([convert(a) for a in index])
        if isinstance(index, ndarray):
            index = index._data
        return ndarray(self._data[index])

    def __eq__(self, other):
        if isinstance(other, (float, complex, int)):
            return self._data == other
        return ndarray(self._data == other._data)

    def __ne__(self, other):
        if isinstance(other, (float, complex, int)):
            return self._data != other
        return ndarray(self._data != other._data)

    def __neg__(self):
        return ndarray(-self._data)

    def __mul__(self, f):
        if isinstance(f, (float, complex)):
            return ndarray(f * self._data)
        return ndarray(f._data * self._data)

    def __rmul__(self, f):
        return ndarray(f * self._data)

    def __imul__(self, f):
        if isinstance(f, (float, complex, int)):
            self._data *= f
        else:
            self._data *= f._data
        return self

    def __truediv__(self, other):
        if isinstance(other, (float, complex, int)):
            return ndarray(self._data / other)
        return ndarray(self._data / other._data)

    def __pow__(self, i: int):
        return ndarray(self._data**i)

    def __add__(self, f):
        if isinstance(f, (float, int, complex)):
            return ndarray(f + self._data)
        return ndarray(f._data + self._data)

    def __sub__(self, f):
        if isinstance(f, float):
            return ndarray(self._data - f)
        return ndarray(self._data - f._data)

    def __rsub__(self, f):
        return ndarray(f - self._data)

    def __radd__(self, f):
        return ndarray(f + self._data)

    def __rtruediv__(self, f):
        return ndarray(f / self._data)

    def __iadd__(self, other):
        if isinstance(other, float):
            self._data += other
        else:
            self._data += other._data
        return self

    def __isub__(self, other):
        if isinstance(other, float):
            self._data -= other
        else:
            self._data -= other._data
        return self

    def __matmul__(self, other):
        return ndarray(self._data @ other._data)

    def ravel(self):
        return ndarray(self._data.ravel())

    def conj(self):
        return ndarray(self._data.conj())

    def reshape(self, shape):
        return ndarray(self._data.reshape(shape))

    def view(self, dtype):
        return ndarray(self._data.view(dtype))

    def item(self):
        return self._data.item()

    def trace(self, offset, axis1, axis2):
        return ndarray(self._data.trace(offset, axis1, axis2))

    def fill(self, val):
        self._data.fill(val)
