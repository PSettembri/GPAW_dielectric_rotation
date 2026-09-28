from __future__ import annotations
import warnings

from typing import Any, Sequence

import numpy as np

parameter_functions = {}

"""
background_charge
external
reuse_wfs_method
"""


def input_parameter(func):
    """Decorator for input-parameter normalization functions."""
    parameter_functions[func.__name__] = func
    return func


def update_dict(default: dict, value: dict | None) -> dict[str, Any]:
    """Create dict with defaults + updates.

    >>> update_dict({'a': 1, 'b': 'hello'}, {'a': 2})
    {'a': 2, 'b': 'hello'}
    >>> update_dict({'a': 1, 'b': 'hello'}, None)
    {'a': 1, 'b': 'hello'}
    >>> update_dict({'a': 1, 'b': 'hello'}, {'c': 2})
    Traceback (most recent call last):
    ValueError: Unknown key: 'c'. Must be one of a, b
    """
    dct = default.copy()
    if value is not None:
        if not value.keys() <= default.keys():
            key = (value.keys() - default.keys()).pop()
            raise ValueError(
                f'Unknown key: {key!r}. Must be one of {", ".join(default)}')
        dct.update(value)
    return dct


class InputParameters:
    basis: Any
    charge: float
    convergence: dict[str, Any]
    eigensolver: dict[str, Any]
    experimental: dict[str, Any]
    external: dict[str, Any]
    gpts: None | Sequence[int]
    h: float | None
    hund: bool
    kpts: dict[str, Any]
    magmoms: Any
    mode: dict[str, Any]
    nbands: None | int | str
    parallel: dict[str, Any]
    poissonsolver: dict[str, Any]
    setups: Any
    soc: bool
    spinpol: bool
    symmetry: dict[str, Any]
    xc: dict[str, Any]

    def __init__(self, params: dict[str, Any], warn: bool = True):
        self.keys = sorted(params)

        for key in params:
            if key not in parameter_functions:
                raise ValueError(
                    f'Unknown parameter {key!r}.  Must be one of: ' +
                    ', '.join(parameter_functions))
        for key, func in parameter_functions.items():
            if key in params:
                param = params[key]
                if hasattr(param, 'todict'):
                    param = param.todict()
                value = func(param)
            else:
                value = func()
            self.__dict__[key] = value

        if self.h is not None and self.gpts is not None:
            raise ValueError("""You can't use both "gpts" and "h"!""")

        if self.experimental is not None:
            if self.experimental.pop('niter_fixdensity', None) is not None:
                warnings.warn('Ignoring "niter_fixdensity".')
            if self.experimental.pop('reuse_wfs_method', None) is not None:
                warnings.warn('Ignoring "reuse_wfs_method".')
            if 'soc' in self.experimental:
                warnings.warn('Please use new "soc" parameter.',
                              DeprecatedParameterWarning)
                self.soc = self.experimental.pop('soc')
            if 'magmoms' in self.experimental:
                warnings.warn('Please use new "magmoms" parameter.',
                              DeprecatedParameterWarning)
                self.magmoms = self.experimental.pop('magmoms')
                self.keys.append('magmoms')
                self.keys.sort()
            assert not self.experimental
            self.keys.remove('experimental')
            self.__dict__.pop('experimental')

        if self.mode.get('name') is None:
            if warn:
                warnings.warn(
                    ('Finite-difference mode implicitly chosen; '
                     'it will be an error to not specify a mode '
                     'in the future'),
                    DeprecatedParameterWarning)
            self.mode = dict(self.mode, name='fd')

    def __repr__(self) -> str:
        p = ', '.join(f'{key}={value!r}'
                      for key, value in self.items())
        return f'InputParameters({p})'

    def items(self):
        for key in self.keys:
            yield key, getattr(self, key)

    def __contains__(self, key):
        return key in self.keys


@input_parameter
def basis(value=None):
    """Atomic basis set."""
    return value or {}


@input_parameter
def charge(value=0.0):
    return value


@input_parameter
def convergence(value=None):
    """Accuracy of the self-consistency cycle."""
    return value or {}


@input_parameter
def eigensolver(value=None) -> dict:
    """Eigensolver."""
    if isinstance(value, str):
        value = {'name': value}
    if value and value['name'] != 'dav':
        warnings.warn(f'{value["name"]} not implemented.  Using dav instead')
        return {'name': 'dav'}
    return value or {}


@input_parameter
def experimental(value=None):
    return value


@input_parameter
def external(value=None):
    return value


@input_parameter
def gpts(value=None):
    """Number of grid points."""
    return value


@input_parameter
def h(value=None):
    """Grid spacing."""
    return value


@input_parameter
def hund(value=False):
    """Using Hund's rule for guessing initial magnetic moments."""
    return value


@input_parameter
def kpts(value=None) -> dict[str, Any]:
    """Brillouin-zone sampling."""
    if value is None:
        value = {'size': (1, 1, 1)}
    elif not isinstance(value, dict):
        array = np.array(value)
        if array.shape == (3,):
            value = {'size': array}
        else:
            value = {'kpts': array}
    return value


@input_parameter
def magmoms(value=None):
    return value


@input_parameter
def maxiter(value=333):
    """Maximum number of SCF-iterations."""
    return value


@input_parameter
def mixer(value=None):
    return value or {}


@input_parameter
def mode(value=None):
    if value is None:
        return {'name': value}
    if isinstance(value, str):
        return {'name': value}
    gc = value.pop('gammacentered', False)
    assert not gc
    return value


@input_parameter
def nbands(value: str | int | None = None) -> str | int | None:
    """Number of electronic bands."""
    return value


@input_parameter
def occupations(value=None):
    return value


@input_parameter
def parallel(value: dict[str, Any] | None = None) -> dict[str, Any]:
    dct = update_dict({'kpt': None,
                       'domain': None,
                       'band': None,
                       'order': 'kdb',
                       'stridebands': False,
                       'augment_grids': False,
                       'sl_auto': False,
                       'sl_default': None,
                       'sl_diagonalize': None,
                       'sl_inverse_cholesky': None,
                       'sl_lcao': None,
                       'sl_lrtddft': None,
                       'use_elpa': False,
                       'elpasolver': '2stage',
                       'buffer_size': None,
                       'gpu': False},
                      value)
    return dct


@input_parameter
def poissonsolver(value=None):
    """Poisson solver."""
    return value or {}


@input_parameter
def random(value=False):
    return value


@input_parameter
def setups(value='paw'):
    """PAW datasets or pseudopotentials."""
    return value if isinstance(value, dict) else {'default': value}


@input_parameter
def soc(value=False):
    return value


@input_parameter
def spinpol(value=False):
    return value


@input_parameter
def symmetry(value='undefined'):
    """Use of symmetry."""
    if value == 'undefined':
        value = {}
    elif value is None or value == 'off':
        value = {'point_group': False, 'time_reversal': False}
    return value


@input_parameter
def xc(value='LDA'):
    """Exchange-Correlation functional."""
    if isinstance(value, str):
        return {'name': value}
    return value


class DeprecatedParameterWarning(FutureWarning):
    """Warning class for when a parameter or its value is deprecated."""
