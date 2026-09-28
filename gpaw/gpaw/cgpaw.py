import _gpaw


def __getattr__(name):
    return getattr(_gpaw, name)
