import numpy as np
import pytest
from ase.build import molecule
from gpaw import GPAW, restart
from gpaw.mom import prepare_mom_calculation


@pytest.mark.mom
def test_mom_fd_energy(in_tmp_dir):
    dE_ref = [7.8009908153, 7.5234341583]

    atoms = molecule('HCl')
    atoms.center(vacuum=2)

    calc = GPAW(mode='fd',
                nbands=6,
                h=0.24,
                xc='PBE',
                spinpol=True,
                convergence={'energy': 100,
                             'density': 1e-3,
                             'eigenstates': 100,
                             'bands': -1})

    atoms.calc = calc
    # Ground-state calculation
    E_gs = atoms.get_potential_energy()

    calc.write('hcl_fd_gs.gpw', 'all')

    # Test spin polarized excited-state calculations
    for s in [0, 1]:
        if s == 1:
            atoms, calc = restart('hcl_fd_gs.gpw', txt='-')

        f_sn = []
        for spin in range(calc.get_number_of_spins()):
            f_n = calc.get_occupation_numbers(spin=spin)
            f_sn.append(f_n)
        f_sn[0][3] -= 1.
        f_sn[s][4] += 1.

        prepare_mom_calculation(calc, atoms, f_sn)

        E_es = atoms.get_potential_energy()

        # Test overlaps
        calc.wfs.occupations.initialize_reference_orbitals()
        for kpt in calc.wfs.kpt_u:
            f_sn = calc.get_occupation_numbers(spin=kpt.s)
            unoccupied = [True for i in range(len(f_sn))]
            P = calc.wfs.occupations.calculate_weights(kpt, 1.0, unoccupied)
            assert (np.allclose(P, f_sn))

        dE = E_es - E_gs
        print(dE)
        assert dE == pytest.approx(dE_ref[s], abs=0.015)

    calc = GPAW(mode='fd',
                nbands=6,
                h=0.24,
                xc='PBE',
                convergence={'energy': 100,
                             'density': 1e-3,
                             'eigenstates': 100,
                             'bands': -1})

    atoms.calc = calc
    # Ground-state calculation
    E_gs = atoms.get_potential_energy()

    # Test spin paired excited-state calculation
    f_n = [calc.get_occupation_numbers(spin=0) / 2.]
    f_n[0][3] -= 0.5
    f_n[0][4] += 0.5

    prepare_mom_calculation(calc, atoms, f_n)
    E_es = atoms.get_potential_energy()

    dE = E_es - E_gs
    print(dE)
    assert dE == pytest.approx(8.4695551944, abs=0.01)
