import pytest
from ase import Atoms
from gpaw import GPAW
from gpaw.utilities.adjust_cell import adjust_cell

h = 0.25
box = 3.0


@pytest.mark.later
def test_spin_spin_contamination_B():
    # B should not have spin contamination
    s = Atoms('B')
    adjust_cell(s, box, h=h)
    s.set_initial_magnetic_moments([-1])

    c = GPAW(mode='fd', xc='LDA', nbands=-3,
             basis='dzp',
             hund=True,
             h=h,
             # mixer=MixerDif(beta=0.05, nmaxold=5, weight=50.0),
             convergence={'eigenstates': 0.078,
                          'density': 5e-3,
                          'energy': 0.1})
    c.calculate(s)

    contamination = min(c.density.get_spin_contamination(s, 0),
                        c.density.get_spin_contamination(s, 1))

    assert contamination == pytest.approx(0.0, abs=0.01)


@pytest.mark.later
def test_spin_spin_contamination_H2():
    # setup H2 at large distance with different spins for the atoms
    s = Atoms('H2', positions=[[0, 0, 0], [0, 0, 3.0]])
    adjust_cell(s, box, h=h)
    s.set_initial_magnetic_moments([-1, 1])

    c = GPAW(mode='fd', xc='LDA',
             nbands=-3,
             h=h,
             convergence={'eigenstates': 0.078,
                          'density': 1e-2,
                          'energy': 0.1})
    c.calculate(s)

    scont_s = [c.density.get_spin_contamination(s),
               c.density.get_spin_contamination(s, 1)]

    assert scont_s[0] == pytest.approx(scont_s[1], abs=2e-4)  # symmetry
    assert scont_s[0] == pytest.approx(0.967, abs=2e-3)
