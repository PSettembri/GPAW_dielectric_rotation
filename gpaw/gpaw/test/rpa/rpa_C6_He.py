import pytest
from ase import Atoms
from gpaw import GPAW, PW
from gpaw.mpi import serial_comm


@pytest.mark.response
@pytest.mark.skip(reason='TODO')
def test_rpa_C6_He():
    from gpaw.xc.rpa_correlation_energy import RPACorrelation
    ecut = 50

    He = Atoms('He')
    He.center(vacuum=1.0)

    calc = GPAW(mode=PW(force_complex_dtype=True),
                xc='PBE',
                communicator=serial_comm)
    He.calc = calc
    He.get_potential_energy()
    calc.diagonalize_full_hamiltonian()

    rpa = RPACorrelation(calc)
    C6_rpa, C6_0 = rpa.get_C6_coefficient(ecut=ecut,
                                          direction=2)

    assert C6_0 == pytest.approx(1.772, abs=0.01)
    assert C6_rpa == pytest.approx(1.387, abs=0.01)
