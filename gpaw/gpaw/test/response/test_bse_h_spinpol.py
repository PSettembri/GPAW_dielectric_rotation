import pytest
import numpy as np
from ase import Atoms
from gpaw import GPAW, PW
from gpaw.response.bse import BSE


@pytest.mark.response
@pytest.mark.serial
def test_bse_spinpol(in_tmp_dir):
    """Make sure spinpolarized eigenvalues in gw_skn work.

    See issue #1066.
    """
    atoms = Atoms('H', magmoms=[1], pbc=True)
    atoms.center(vacuum=1.5)
    atoms.calc = GPAW(mode=PW(180, force_complex_dtype=True),
                      nbands=6,
                      convergence={'bands': 4})
    atoms.get_potential_energy()

    gw_skn = np.zeros((2, 1, 2))
    gw_skn[0, 0] = [-10, 2]
    gw_skn[1, 0] = [2, 4]

    bse = BSE(atoms.calc,
              ecut=10,
              nbands=2,
              gw_skn=gw_skn,
              valence_bands=[[0], [0]],
              conduction_bands=[[1], [1]])

    bsematrix = bse.get_bse_matrix()
    w_T, _ = bse.diagonalize_bse_matrix(bsematrix)
    assert w_T[0] == pytest.approx(0.013, abs=0.001)
