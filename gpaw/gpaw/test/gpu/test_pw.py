import pytest
from ase import Atoms
from ase.units import Ha

from gpaw.new.calculation import DFTCalculation
from gpaw.mpi import size
from gpaw.poisson import FDPoissonSolver
from gpaw.new.c import GPU_AWARE_MPI


@pytest.mark.gpu
@pytest.mark.serial
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('gpu', [False, True])
@pytest.mark.parametrize('mode', ['pw', 'fd'])
def test_gpu(dtype, gpu, mode):
    atoms = Atoms('H2')
    atoms.positions[1, 0] = 0.75
    atoms.center(vacuum=1.0)
    if mode == 'fd':
        poisson = FDPoissonSolver()
        h = 0.17
    else:
        poisson = None
        h = None
    dft = DFTCalculation.from_parameters(
        atoms,
        dict(mode={'name': mode,
                   'force_complex_dtype': dtype == complex},
             poissonsolver=poisson,
             h=h,
             convergence={'density': 1e-8},
             parallel={'gpu': gpu},
             setups='paw'),
        log='-')
    dft.converge()
    dft.energies()
    energy = dft.results['energy'] * Ha
    if mode == 'pw':
        assert energy == pytest.approx(-16.032945, abs=1e-6)
    else:
        assert energy == pytest.approx(5.071972893296197, abs=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(size > 2, reason='Not implemented')
@pytest.mark.parametrize('gpu', [False, True])
@pytest.mark.parametrize('par', ['domain', 'kpt', 'band'])
@pytest.mark.parametrize('mode', ['pw', 'fd'])
@pytest.mark.parametrize('xc', ['LDA', 'PBE'])
def test_gpu_k(gpu, par, mode, xc):
    if gpu and size > 1 and not GPU_AWARE_MPI:
        if mode == 'fd' and par == 'domain':
            pytest.skip('Domain decomposition needs GPU-aware MPI')
        if mode == 'pw' and par == 'domain' and xc == 'PBE':
            pytest.skip('Domain decomposition needs GPU-aware MPI')
        if mode == 'fd':
            pytest.skip('???')
        if mode == 'pw' and par in ['kpt', 'band'] and xc == 'PBE':
            pytest.skip('???')

    if gpu and size > 1:
        if mode == 'fd' and par == 'band':
            pytest.skip('FAILING')

    if size == 1 and par in ['kpt', 'band']:
        pytest.skip('Not testing parallelization on single core')

    atoms = Atoms('H', pbc=True, cell=[1, 1.1, 1.1])
    if mode == 'fd':
        poisson = FDPoissonSolver()
        h = 0.09
    else:
        poisson = None
        h = None

    dft = DFTCalculation.from_parameters(
        atoms,
        dict(mode={'name': mode},
             spinpol=True,
             xc=xc,
             h=h,
             convergence={'density': 1e-8},
             kpts=(4, 1, 1),
             poissonsolver=poisson,
             parallel={'gpu': gpu,
                       par: size},
             setups='paw'),
        log='-')
    dft.converge()
    dft.energies()
    if mode == 'pw':
        dft.forces()
        dft.stress()
    energy = dft.results['energy'] * Ha
    ref = {'LDAfd': -17.685022604078714,
           'PBEfd': -17.336991943070384,
           'PBEpw': -17.304186,
           'LDApw': -17.653433}[xc + mode]
    assert energy == pytest.approx(ref, abs=1e-6)
