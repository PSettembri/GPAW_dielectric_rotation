from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


def obj2yaml(obj: Any, indentation: str = '') -> str:
    """Convert Python object to YAML string.

    >>> print(obj2yaml({'a': {'b': 42}}))
    a:
      b: 42
    """
    if isinstance(obj, dict):
        i = indentation
        txt = f'\n{i}'.join(f'{k}: {obj2yaml(v, i + "  ")}'
                            for k, v in obj.items())
        if i:
            return '\n' + i + txt
        return txt.replace(': \n', ':\n')
    return repr(obj)


def indent(text: Any, indentation='  ') -> str:
    r"""Indent text blob.

    >>> indent('line 1\nline 2', '..')
    '..line 1\n..line 2'
    """
    if not isinstance(text, str):
        text = str(text)
    return indentation + text.replace('\n', '\n' + indentation)


def comment(text: Any) -> str:
    """Comment text blob.

    >>> comment('hmm')
    '# hmm'
    """
    return indent(text, '# ')


def read_gpaw_yaml(fd, index):
    r"""Convert YAML text output from GPAW calculation to Atoms object(s).

    >>> import io
    >>> y = ('atoms: [[H, [0, 0, 0], [0, 0, 1]]]\n'
    ...      'cell: [[0, 0, 0], [0, 0, 0], [0, 0, 0]]\n'
    ...      'periodic: [false, false, false]\n')
    >>> read_gpaw_yaml(io.StringIO(y), 0)
    Atoms(symbols='H', pbc=False)
    """
    import yaml
    configs = []
    for dct in yaml.safe_load_all(fd):
        if 'atoms' in dct:
            atoms = dict2atoms(dct)
            configs.append(atoms)
    return configs[index]


def dict2atoms(dct) -> Atoms:
    symbols = []
    positions = []
    magmoms = []
    for symbol, position, (_, _, magmom) in dct['atoms']:
        symbols.append(symbol)
        positions.append(position)
        magmoms.append(magmom)

    cell = dct['cell']
    pbc = dct['periodic']

    atoms = Atoms(symbols,
                  positions,
                  cell=cell,
                  pbc=pbc)

    if 'energies' in dct:
        energy = dct['energies']['extrapolated']
        if 'forces' in dct:
            forces = dct['forces']
        else:
            forces = None
        if 'stress tensor' in dct:
            stress = np.array(dct['stress tensor']).ravel()[0, 4, 8, 5, 2, 1]
        else:
            stress = None
        atoms.calc = SinglePointCalculator(energy=energy,
                                           forces=forces,
                                           stress=stress,
                                           atoms=atoms)
    return atoms


if __name__ == '__main__':
    import sys
    import pprint
    import yaml
    for dct in yaml.safe_load_all(open(sys.argv[1])):
        print('---')
        pprint.pp(dct)
