import warnings
import pytest

from ase.build import molecule
from gpaw import GPAW

# Silence those warnings.


@pytest.mark.legacy
def test_fileio_idiotproof_setup(in_tmp_dir):
    warnings.filterwarnings('ignore', 'Setup for',)

    m = molecule('H')
    m.center(vacuum=2.0)
    calc = GPAW(mode='lcao')
    m.calc = calc
    m.get_potential_energy()
    calc.write('r.gpw')
    calc = GPAW('r.gpw', xc='PBE')
    calc.get_potential_energy()
