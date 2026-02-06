import numpy as np
from gpaw.external import ConstantElectricField
from ase.units import alpha, Hartree, Bohr
from gpaw.lcaotddft.hamiltonian import KickHamiltonian


class RRemission:
    r"""
    Radiation-reaction potential accoridng to Schaefer et al.
    [arXiv 2109.09839] The potential accounts for the friction
    forces acting on the radiating system of oscillating charges
    emitting into a single dimension. A more elegant
    formulation would use the current instead of the dipole.
    Please contact christian.schaefer.physics@gmail.com if any problems
    should appear or you would like to consider more complex emission.
    Big thanks to Tuomas Rossi and Jakub Fojt for their help.

    Parameters
    ----------
    rr_quantization_plane: float
        value of :math:`rr_quantization_plane` in atomic units
    pol_cavity: array
        value of :math:`pol_cavity` dimensionless (directional)
    """

    def __init__(self, rr_quantization_plane_in, pol_cavity_in):
        self.rr_quantization_plane = rr_quantization_plane_in / Bohr**2
        self.polarization_cavity = pol_cavity_in

    def initialize(self, paw):
        self.iterpredcop = 0
        self.time_previous = paw.time
        self.dipolexyz = [0, 0, 0]
        self.density = paw.density
        self.wfs = paw.wfs
        self.hamiltonian = paw.hamiltonian
        self.dipolexyz_previous = self.density.calculate_dipole_moment()

    def vradiationreaction(self, kpt, time):
        if self.iterpredcop == 0:
            self.iterpredcop += 1
            self.dipolexyz_previous = self.density.calculate_dipole_moment()
            self.time_previous = time
            deltat = 1
        else:
            self.iterpredcop = 0
            deltat = time - self.time_previous
            self.dipolexyz = (self.density.calculate_dipole_moment()
                              - self.dipolexyz_previous) / deltat
        # function uses V/Angstroem and therefore conversion necessary
        ext = ConstantElectricField(Hartree / Bohr, self.polarization_cavity)
        uvalue = 0
        self.ext_i = []
        self.ext_i.append(ext)
        get_matrix = self.wfs.eigensolver.calculate_hamiltonian_matrix
        self.V_iuMM = []
        for ext in self.ext_i:
            V_uMM = []
            hamiltonian = KickHamiltonian(self.hamiltonian, self.density, ext)
            for kpt in self.wfs.kpt_u:
                V_MM = get_matrix(hamiltonian, self.wfs, kpt,
                                  add_kinetic=False, root=-1)
                V_uMM.append(V_MM)
            self.V_iuMM.append(V_uMM)
        self.Ni = len(self.ext_i)
        rr_argument = (-4.0 * np.pi * alpha / self.rr_quantization_plane
                       * np.dot(self.polarization_cavity, self.dipolexyz))
        Vrr_MM = rr_argument * self.V_iuMM[0][uvalue]
        for i in range(1, self.Ni):
            Vrr_MM += rr_argument * self.V_iuMM[i][uvalue]
        return Vrr_MM
