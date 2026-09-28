import pytest
from ase import Atoms
from gpaw import GPAW
from gpaw.lcao.scissors import Scissors


@pytest.mark.later
def test_scissors():
    h2 = Atoms('2H2', [[0, 0, 0], [0, 0, 0.74],
                       [4, 0, 0], [4, 0, 0.74]])
    h2.center(vacuum=3.0)
    d = 1.0
    h2.calc = GPAW(mode='lcao',
                   basis='sz(dzp)',
                   eigensolver=Scissors([(-d, d, 2)]),
                   txt=None)
    h2.get_potential_energy()
    e1, e2, e3, e4 = h2.calc.get_eigenvalues()
    assert e2 - e1 == pytest.approx(d, abs=0.01)
    assert e4 - e3 == pytest.approx(d, abs=0.01)
