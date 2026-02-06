"""Make sure we get a warning when mode is not supplied."""
from ase.build import molecule
from gpaw.calculator import (GPAW as OldGPAW,
                             DeprecatedParameterWarning as OldDPW)
from gpaw.new.ase_interface import GPAW as NewGPAW
from gpaw.new.input_parameters import DeprecatedParameterWarning as NewDPW
import pytest


@pytest.mark.ci
@pytest.mark.parametrize('new', [True, False])
def test_no_mode_supplied(new: bool) -> None:
    if new:
        GPAW, DPWarning = NewGPAW, NewDPW
    else:
        GPAW, DPWarning = OldGPAW, OldDPW
    a = 6.0
    hydrogen = molecule('H2', cell=[a, a, a])
    hydrogen.center()
    with pytest.warns(DPWarning):
        hydrogen.calc = GPAW()
        hydrogen.calc.initialize(hydrogen)
