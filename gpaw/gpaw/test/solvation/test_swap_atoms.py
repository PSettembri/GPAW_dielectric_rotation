from gpaw.utilities.adjust_cell import adjust_cell
from ase.build import molecule
from ase.units import Pascal, m
from ase.data.vdw import vdw_radii
from gpaw.solvation import (
    SolvationGPAW,
    EffectivePotentialCavity,
    Power12Potential,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction)


vdw_radii = vdw_radii.copy()


def test_solvation_swap_atoms():
    h = 0.3
    vac = 3.0
    u0 = 0.180
    epsinf = 80.
    st = 18.4 * 1e-3 * Pascal * m
    T = 298.15
    vdw_radii[1] = 1.09

    def atomic_radii(atoms):
        return [vdw_radii[n] for n in atoms.numbers]

    convergence = {
        'energy': 0.1 / 8.,
        'density': 10.,
        'eigenstates': 10.,
    }

    atoms = molecule('H2O')
    adjust_cell(atoms, vac, h)

    calc = SolvationGPAW(
        mode='fd', xc='LDA', h=h, convergence=convergence,
        cavity=EffectivePotentialCavity(
            effective_potential=Power12Potential(atomic_radii, u0),
            temperature=T,
            surface_calculator=GradientSurface()
        ),
        dielectric=LinearDielectric(epsinf=epsinf),
        interactions=[SurfaceInteraction(surface_tension=st)]
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.get_forces()
    eps_gradeps = calc.hamiltonian.dielectric.eps_gradeps

    # same molecules, different cell, reallocate
    atoms = molecule('H2O')
    atoms.positions[0][0] = atoms.positions[0][0] - 1.
    adjust_cell(atoms, vac, h)
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.get_forces()
    assert calc.hamiltonian.dielectric.eps_gradeps is not eps_gradeps
    eps_gradeps = calc.hamiltonian.dielectric.eps_gradeps

    # small position change, no reallocate
    atoms.positions[0][0] = atoms.positions[0][0] + 1e-2
    atoms.get_potential_energy()
    atoms.get_forces()
    assert calc.hamiltonian.dielectric.eps_gradeps is eps_gradeps
    eps_gradeps = calc.hamiltonian.dielectric.eps_gradeps
    radii = calc.hamiltonian.cavity.effective_potential.r12_a

    # completely different atoms object, reallocate, read new radii
    atoms = molecule('NH3')
    adjust_cell(atoms, vac, h)
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.get_forces()
    assert calc.hamiltonian.dielectric.eps_gradeps is not eps_gradeps
    assert calc.hamiltonian.cavity.effective_potential.r12_a is not radii
