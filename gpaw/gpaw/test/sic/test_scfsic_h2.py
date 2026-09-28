import pytest
from ase import Atoms

from gpaw import GPAW, restart


@pytest.mark.later
def test_sic_scfsic_h2(in_tmp_dir):
    a = 6.0
    atom = Atoms('H', magmoms=[1.0], cell=(a, a, a))
    molecule = Atoms('H2', positions=[
                     (0, 0, 0), (0, 0, 0.737)], cell=(a, a, a))
    atom.center()
    molecule.center()

    calc = GPAW(mode='fd',
                xc='LDA-PZ-SIC',
                eigensolver='rmm-diis',
                txt='h2.sic.txt',
                setups='hgh')

    atom.calc = calc
    atom.get_potential_energy()

    molecule.calc = calc
    e2 = molecule.get_potential_energy()
    molecule.get_forces()
    # de = 2 * e1 - e2
    # assert de == pytest.approx(4.5, abs=0.1)

    # Test forces ...

    calc.write('H2.gpw', mode='all')
    atoms, calc = restart('H2.gpw')
    e2b = atoms.get_potential_energy()
    assert e2 == pytest.approx(e2b, abs=0.0001)
