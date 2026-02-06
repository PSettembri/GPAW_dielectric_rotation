import pytest
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from gpaw.jellium import JelliumSlab
from gpaw import GPAW, Mixer


@pytest.mark.later
@pytest.mark.libxc
def test_jellium(in_tmp_dir):
    rs = 5.0 * Bohr  # Wigner-Seitz radius
    h = 0.24          # grid-spacing
    a = 8 * h        # lattice constant
    v = 3 * a        # vacuum
    L = 8 * a       # thickness
    k = 6           # number of k-points (k*k*1)

    ne = a**2 * L / (4 * np.pi / 3 * rs**3)

    x = h / 4  # make sure surfaces are between grid-points
    bc = JelliumSlab(ne, z1=v - x, z2=v + L + x)

    surf = Atoms(pbc=(True, True, False),
                 cell=(a, a, v + L + v))
    surf.calc = GPAW(mode='fd',
                     background_charge=bc,
                     poissonsolver={'dipolelayer': 'xy'},
                     xc='LDA_X+LDA_C_WIGNER',
                     eigensolver='dav',
                     kpts=[k, k, 1],
                     h=h,
                     maxiter=300,
                     convergence={'density': 1e-5},
                     mixer=Mixer(0.3, 7, 100),
                     nbands=int(ne / 2) + 15,
                     txt='surface.txt')
    _ = surf.get_potential_energy()

    efermi = surf.calc.get_fermi_level()
    # Get (x-y-averaged) electrostatic potential
    # Must collect it from the CPUs
    # https://listserv.fysik.dtu.dk/pipermail/gpaw-users/2014-January/002524.html
    ham = surf.calc.hamiltonian
    v = (ham.finegd.collect(ham.vHt_g,
                            broadcast=True) * Hartree).mean(0).mean(0)

    # Get the work function
    phi1 = v[-1] - efermi

    assert phi1 == pytest.approx(2.715, abs=1e-3)
    # Reference value: Lang and Kohn, 1971, Theory of Metal Surfaces:
    # Work function
    # DOI 10.1103/PhysRevB.3.1215
    # r_s = 5, work function = 2.73 eV

    surf.calc.write('jellium.gpw')
