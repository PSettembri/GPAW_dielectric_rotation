import time

import pytest
import numpy as np
from ase.build import bulk
from ase.parallel import parprint

from gpaw import GPAW, PW
from gpaw.test import findpeak
from gpaw.response.df import DielectricFunction, read_response_function
from gpaw.mpi import size, world


@pytest.mark.dielectricfunction
@pytest.mark.response
@pytest.mark.libxc
def test_response_aluminum_EELS_ALDA(in_tmp_dir):
    assert size <= 4**3

    # Ground state calculation

    t1 = time.time()

    a = 4.043
    atoms = bulk('Al', 'fcc', a=a)
    atoms.center()
    calc = GPAW(mode=PW(200),
                nbands=4,
                kpts=(4, 4, 4),
                parallel={'band': 1},
                xc='LDA')

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('Al', 'all')
    t2 = time.time()

    # Excited state calculation
    q = np.array([1 / 4, 0, 0])
    w = np.linspace(0, 24, 241)

    df = DielectricFunction(calc='Al', frequencies=w, eta=0.2, ecut=50,
                            hilbert=False)
    df.get_eels_spectrum(xc='ALDA', filename='EELS_Al_ALDA.csv', q_c=q)

    t3 = time.time()

    parprint('For ground  state calc, it took', (t2 - t1) / 60, 'minutes')
    parprint('For excited state calc, it took', (t3 - t2) / 60, 'minutes')

    world.barrier()
    omega_w, eels0_w, eels_w = read_response_function('EELS_Al_ALDA.csv')

    # New results are compared with test values
    wpeak1, Ipeak1 = findpeak(omega_w, eels0_w)
    wpeak2, Ipeak2 = findpeak(omega_w, eels_w)

    test_wpeak1 = 15.7002862696  # eV
    test_Ipeak1 = 28.5590363176
    test_wpeak2 = 15.5196459206
    test_Ipeak2 = 26.5680624522

    if abs(test_wpeak1 - wpeak1) > 0.02 or abs(test_wpeak2 - wpeak2) > 0.02:
        print(test_wpeak1 - wpeak1, test_wpeak2 - wpeak2)
        raise ValueError('Plasmon peak not correct ! ')

    if abs(test_Ipeak1 - Ipeak1) > 1 or abs(test_Ipeak2 - Ipeak2) > 1:
        print(Ipeak1 - test_Ipeak1, Ipeak2 - test_Ipeak2)
        raise ValueError('Please check spectrum strength ! ')
