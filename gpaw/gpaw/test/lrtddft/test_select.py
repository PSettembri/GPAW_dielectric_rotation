import pytest
from ase import Atoms
from gpaw import GPAW, FermiDirac
from gpaw.lrtddft import LrTDDFT


@pytest.mark.lrtddft
def test_digonalize():
    """Test selection at diagonalization stage"""
    atoms = Atoms('O')
    atoms.cell = [3, 4, 5]
    atoms.center()

    atoms.calc = GPAW(mode='fd',
                      occupations=FermiDirac(width=0.1),
                      nbands=5)
    atoms.get_potential_energy()

    lr = LrTDDFT(atoms.calc)

    # all
    lr.diagonalize()
    assert len(lr) == 10

    lr.diagonalize(restrict={'istart': 3})
    assert len(lr) == 1

    lr.diagonalize(restrict={'jend': 1})
    assert len(lr) == 1

    lr.diagonalize(restrict={'eps': 1.5})
    assert len(lr) == 2

    lr.diagonalize(restrict={'energy_range': 1})
    assert len(lr) == 3
