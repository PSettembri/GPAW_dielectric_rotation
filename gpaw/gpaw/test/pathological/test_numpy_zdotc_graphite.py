"""gpaw-python Segmentation faults
when gpaw-python and numpy are linked to different blas"""
from math import sqrt

import pytest
from ase import Atoms

from gpaw import GPAW, ConvergenceError

kpts = (2, 1, 1)
a = 1.42
c = 3.355


@pytest.mark.legacy
def test_pathological_numpy_zdotc_graphite():
    # AB stack
    atoms = Atoms('C4',
                  [(1 / 3, 1 / 3, 0),
                   (2 / 3, 2 / 3, 0),
                   (0, 0, 0.5),
                   (1 / 3, 1 / 3, 0.5)],
                  pbc=(1, 1, 1))

    atoms.set_cell([(sqrt(3) * a / 2, 3 / 2 * a, 0),
                    (-sqrt(3) * a / 2, 3 / 2 * a, 0),
                    (0, 0, 2 * c)],
                   scale_atoms=True)

    calc = GPAW(mode='fd', gpts=(8, 8, 20), nbands=9, kpts=kpts, maxiter=1)

    atoms.calc = calc

    try:
        atoms.get_potential_energy()
    except ConvergenceError:
        pass
