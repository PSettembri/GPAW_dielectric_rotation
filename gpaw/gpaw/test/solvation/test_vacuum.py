import numpy as np
from ase.build import molecule
from ase.data.vdw import vdw_radii
from gpaw import GPAW
from gpaw.utilities.adjust_cell import adjust_cell
import pytest
from gpaw.solvation import (SolvationGPAW, EffectivePotentialCavity,
                            Power12Potential, LinearDielectric)

vdw_radii = vdw_radii.copy()


def test_solvation_vacuum():

    SKIP_REF_CALC = True

    energy_eps = 0.0005 / 8.
    forces_eps = 3e-2

    h = 0.3
    vac = 3.0
    u0 = .180
    T = 298.15
    vdw_radii[1] = 1.09

    def atomic_radii(atoms):
        return [vdw_radii[n] for n in atoms.numbers]

    atoms = molecule('H2O')
    adjust_cell(atoms, vac, h)

    convergence = {
        'energy': energy_eps,
        'forces': forces_eps ** 2,  # Force error is squared
        'density': 10.,
        'eigenstates': 10.,
    }

    if not SKIP_REF_CALC:
        atoms.calc = GPAW(mode='fd', xc='LDA', h=h, convergence=convergence)
        Eref = atoms.get_potential_energy()
        print(Eref)
        Fref = atoms.get_forces()
        print(Fref)
    else:
        # setups: 0.9.11271, same settings as above
        Eref = -11.9932

        Fref = np.array(
            [[1.95122040e-12, -1.17770462e-12, -6.04993798e+00],
             [6.61270337e-14, 1.58227909e+00, 6.06605145e-02],
             [1.35947527e-13, -1.58227909e+00, 6.06605145e-02]])

    atoms.calc = SolvationGPAW(
        mode='fd', xc='LDA', h=h, convergence=convergence,
        cavity=EffectivePotentialCavity(
            effective_potential=Power12Potential(atomic_radii=atomic_radii,
                                                 u0=u0),
            temperature=T
        ),
        dielectric=LinearDielectric(epsinf=1.0),
    )
    Etest = atoms.get_potential_energy()
    Eeltest = atoms.calc.get_electrostatic_energy()
    Ftest = atoms.get_forces()
    assert Etest == pytest.approx(
        Eref, abs=energy_eps * atoms.calc.get_number_of_electrons())
    assert Ftest == pytest.approx(Fref, abs=forces_eps)
    assert Eeltest == pytest.approx(Etest, abs=0.0)
