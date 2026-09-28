import numpy as np
import pytest
from ase import Atoms
from ase.phonons import Phonons

from gpaw import GPAW
from gpaw.elph.electronphonon import ElectronPhononCoupling
from gpaw.mpi import world

pytestmark = pytest.mark.skipif(world.size > 2,
                                reason='world.size > 2')


@pytest.mark.elph
def test_electronphonon(in_tmp_dir):
    a = 0.90
    atoms = Atoms('H',
                  cell=np.diag([a, 2.1, 2.1]),
                  positions=[[0, 0, 0]],
                  pbc=(1, 0, 0))

    atoms.center()
    supercell = (2, 1, 1)
    parameters = {'mode': 'lcao',
                  'kpts': {'size': (2, 1, 1), 'gamma': True},
                  'txt': None,
                  'basis': 'dzp',
                  'symmetry': {'point_group': False},
                  'parallel': {'domain': 1},
                  'xc': 'PBE'}
    elph_calc = GPAW(**parameters)
    atoms.calc = elph_calc
    atoms.get_potential_energy()
    gamma_bands = elph_calc.wfs.kpt_u[0].C_nM

    elph = ElectronPhononCoupling(atoms, elph_calc, supercell=supercell,
                                  name='elph+ph', calculate_forces=True)
    elph.run()

    elph.set_lcao_calculator(elph_calc)
    elph.calculate_supercell_matrix()

    ph = Phonons(atoms=atoms, name='elph+ph', supercell=supercell, calc=None)
    ph.read()
    kpts = [[0, 0, 0]]
    frequencies, modes = ph.band_structure(kpts, modes=True)

    c_kn = np.array([[gamma_bands[0]]])
    _ = elph.bloch_matrix(c_kn=c_kn, kpts=kpts, qpts=kpts, u_ql=modes)
