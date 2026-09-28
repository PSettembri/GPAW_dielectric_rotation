import numpy as np
import pytest
from ase import Atoms
from gpaw import PW, FermiDirac
from gpaw.new.ase_interface import GPAW as NewGPAW
from gpaw import GPAW as AnyGPAW
from gpaw.mpi import world

# Domain and k-point parallelization:
dk = []
for d in [1, 2, 4, 8]:
    for k in [1, 2]:
        if d * k > world.size:
            continue
        dk.append((d, k))


@pytest.mark.stress
@pytest.mark.parametrize('d, k', dk)
@pytest.mark.parametrize(
    'gpu', [False,
            pytest.param(
                True,
                marks=[pytest.mark.gpu])])
def test_pw_par_strategies(in_tmp_dir, d, k, gpu, gpaw_new):
    if (gpu or gpaw_new) and d > 1:
        pytest.skip()

    ecut = 200
    kpoints = [1, 1, 4]
    atoms = Atoms('HLi',
                  cell=[6, 6, 3.4],
                  pbc=True,
                  positions=[[3, 3, 0],
                             [3, 3, 1.6]])
    parallel = {'domain': d, 'kpt': k}
    if gpu:
        parallel['gpu'] = True
        GPAW = NewGPAW
    else:
        GPAW = AnyGPAW
    atoms.calc = GPAW(mode=PW(ecut),
                      txt='hli.txt',
                      parallel=parallel,
                      kpts={'size': kpoints},
                      convergence={'maximum iterations': 4},
                      occupations=FermiDirac(width=0.1))

    e = atoms.get_potential_energy()
    assert e == pytest.approx(-5.218064604018109, abs=1e-9)

    f = atoms.get_forces()
    assert f == pytest.approx(np.array([[0, 0, -7.85130336e-01],
                                        [0, 0, 8.00667631e-01]]))

    s = atoms.get_stress()
    assert s == pytest.approx(
        [3.98105501e-03, 3.98105501e-03, -4.98044912e-03, 0, 0, 0])

    atoms.calc.write('hli.gpw', mode='all')
    GPAW('hli.gpw', txt=None)
