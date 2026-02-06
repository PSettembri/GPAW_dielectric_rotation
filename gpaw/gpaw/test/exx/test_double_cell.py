import pytest
from ase import Atoms
from gpaw import GPAW, PW


@pytest.mark.libxc
@pytest.mark.hybrids
def test_exx_double_cell(in_tmp_dir):
    L = 2.6
    a = Atoms('H2',
              [[0, 0, 0], [0.5, 0.5, 0]],
              cell=[L, L, 1],
              pbc=1)
    a.center()

    a.calc = GPAW(
        mode=PW(400),
        symmetry='off',
        kpts={'size': (1, 1, 4), 'gamma': True},
        convergence={'density': 1e-6},
        spinpol=True,
        txt='H2.txt',
        xc='HSE06')
    e1 = a.get_potential_energy()
    eps1 = a.calc.get_eigenvalues(1)[0]
    f1 = a.get_forces()
    assert abs(f1[1, 0] - 9.60644) < 0.0005
    if 0:
        # To check against numeric calculation of the forces, but it takes
        # more time
        from ase.calculators.test import numeric_force
        f1n = numeric_force(a, 1, 0, 0.001)
        assert abs(f1[1, 0] - f1n) < 0.0005

    a *= (1, 1, 2)
    a.calc = GPAW(
        mode=PW(400),
        kpts={'size': (1, 1, 2), 'gamma': True},
        symmetry='off',
        convergence={'density': 1e-6},
        txt='H4.txt',
        xc='HSE06')
    e2 = a.get_potential_energy()
    eps2 = a.calc.get_eigenvalues(0)[0]
    f2 = a.get_forces()

    f2[:2] -= f1
    f2[2:] -= f1

    assert abs(e2 - 2 * e1) < 0.002
    assert abs(eps1 - eps2) < 0.001
    assert abs(f2).max() < 0.00085


if __name__ == '__main__':
    from cProfile import Profile
    prof = Profile()
    prof.enable()
    test_exx_double_cell(1)
    prof.disable()
    from gpaw.mpi import rank, size
    prof.dump_stats(f'prof-{size}.{rank}')
