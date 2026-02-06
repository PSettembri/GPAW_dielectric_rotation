# Test that the atomic corrections of LCAO work correctly,
# by verifying that the different implementations yield the same numbers.
#
# For example the corrections P^* dH P to the Hamiltonian.
#
# This is done by invoking GPAW once for each type of calculation.

from itertools import count

import pytest
from ase.build import molecule, bulk

from gpaw import GPAW, LCAO
from gpaw.mpi import world


def system1():
    system = molecule('CH3CH2OH')
    system.center(vacuum=3.0)
    system.pbc = (0, 1, 1)
    system = system.repeat((1, 1, 2))
    system.rattle(stdev=0.05)
    return system


def system2():
    return bulk('Cu', orthorhombic=True) * (2, 1, 2)


@pytest.mark.later
@pytest.mark.parametrize('atoms, kpts', [
    (system1(), [1, 1, 1]),
    (system2(), [2, 3, 4]),
])
def test_lcao_atomic_corrections(atoms, in_tmp_dir, scalapack, kpts):
    # Use a cell large enough that some overlaps are zero.
    # Thus the matrices will have at least some sparsity.

    corrections = ['dense', 'sparse']

    counter = count()
    energies = []
    for correction in corrections:
        parallel = {}
        if world.size >= 4:
            parallel['band'] = 2
            # if correction.name != 'dense':
            parallel['sl_auto'] = True
        calc = GPAW(mode=LCAO(atomic_correction=correction),
                    basis='sz(dzp)',
                    # spinpol=True,
                    parallel=parallel,
                    txt='gpaw.{}.txt'.format(next(counter)),
                    h=0.35, kpts=kpts,
                    convergence={'maximum iterations': 2})
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        energies.append(energy)
        if calc.world.rank == 0:
            print('e', energy)

    master = calc.wfs.world.rank == 0
    if master:
        print('energies', energies)

    eref = energies[0]
    errs = []
    for energy, c in zip(energies, corrections):
        err = abs(energy - eref)
        errs.append(err)
        if master:
            print('err=%e :: name=%s' % (err, correction))

    maxerr = max(errs)
    assert maxerr < 1e-11, maxerr
