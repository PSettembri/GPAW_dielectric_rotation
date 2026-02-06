import pytest

from gpaw import GPAW, PW, restart
from ase import Atoms
import numpy as np
from gpaw.directmin.etdm_fdpw import FDPWETDM


@pytest.mark.later
@pytest.mark.sic
def test_pz_localization_pw(in_tmp_dir):
    """
    Test Perdew-Zunger and Kohn-Sham localizations in PW mode
    :param in_tmp_dir:
    :return:
    """

    # Water molecule:
    d = 0.9575
    t = np.pi / 180 * (104.51 + 2.0)
    eps = 0.02
    H2O = Atoms('OH2',
                positions=[(0, 0, 0),
                           (d + eps, 0, 0),
                           (d * np.cos(t), d * np.sin(t), 0)])
    H2O.center(vacuum=3.0)

    calc = GPAW(mode=PW(300, force_complex_dtype=True),
                occupations={'name': 'fixed-uniform'},
                convergence={'energy': np.inf,
                             'eigenstates': np.inf,
                             'density': np.inf,
                             'minimum iterations': 0},
                eigensolver=FDPWETDM(converge_unocc=False),
                mixer={'backend': 'no-mixing'},
                symmetry='off',
                spinpol=True
                )
    H2O.calc = calc
    H2O.get_potential_energy()
    calc.write('h2o.gpw', mode='all')

    H2O, calc = restart('h2o.gpw', txt='-')

    calc.set(eigensolver=FDPWETDM(
             functional={'name': 'PZ-SIC',
                         'scaling_factor': (0.5, 0.5)},
             localizationseed=42,
             localizationtype='KS_PZ',
             localization_tol=5.0e-2,
             converge_unocc=False))
    e = H2O.get_potential_energy()
    assert e == pytest.approx(-10.118236, abs=0.1)
