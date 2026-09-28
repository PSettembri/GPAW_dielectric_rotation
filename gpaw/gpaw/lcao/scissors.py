"""Scissors operator for LCAO."""
from typing import Sequence, Tuple

import numpy as np
from ase.units import Ha

from gpaw.hamiltonian import Hamiltonian
from gpaw.typing import Array2D, Array3D
from gpaw.kpoint import KPoint
from gpaw.wavefunctions.base import WaveFunctions

from .eigensolver import DirectLCAO


class Scissors(DirectLCAO):
    def __init__(self, shifts: Sequence[Tuple[float, float, int]]):
        """Scissors-operator eigensolver.

        The *shifts* are given as a sequence of tuples::

            [(<shift for occupied states>,
              <shift for unoccupied states>,
              <number of atoms>),
             ...]

        Here we open a gap for states on atoms with indices 3, 4 and 5:

        >>> eigensolver = Scissors([(0.0, 0.0, 3),
        ...                         (-0.5, 0.5, 3)])

        """
        DirectLCAO.__init__(self)
        self.shifts = []
        for homo, lumo, natoms in shifts:
            self.shifts.append((homo / Ha, lumo / Ha, natoms))

    def write(self, writer):
        writer.write(name='lcao')

    def __repr__(self):
        txt = DirectLCAO.__repr__(self)
        txt += '\n    Scissors operators:\n'
        a1 = 0
        for homo, lumo, natoms in self.shifts:
            a2 = a1 + natoms
            txt += (f'      Atoms {a1}-{a2 - 1}: '
                    f'VB: {homo * Ha:+.3f} eV, '
                    f'CB: {lumo * Ha:+.3f} eV\n')
            a1 = a2
        return txt

    def calculate_hamiltonian_matrix(self,
                                     ham: Hamiltonian,
                                     wfs: WaveFunctions,
                                     kpt: KPoint,
                                     Vt_xMM: Array3D = None,
                                     root: int = -1,
                                     add_kinetic: bool = True) -> Array2D:
        """Add scissors operator."""
        H_MM = DirectLCAO.calculate_hamiltonian_matrix(
            self, ham, wfs, kpt, Vt_xMM, root, add_kinetic)
        if kpt.C_nM is None:
            return H_MM

        S_MM = wfs.S_qMM[kpt.q]
        assert abs(S_MM - S_MM.T.conj()).max() < 1e-10

        nocc = ham.setups.nvalence // 2
        C_nM = kpt.C_nM
        iC_Mn = np.linalg.inv(C_nM)
        Co_nM = C_nM.copy()
        Co_nM[nocc:] = 0.0
        Cu_nM = C_nM.copy()
        Cu_nM[:nocc] = 0.0

        M1 = 0
        a1 = 0
        for homo, lumo, natoms in self.shifts:
            a2 = a1 + natoms
            M2 = M1 + sum(setup.nao for setup in ham.setups[a1:a2])

            D_MM = np.zeros_like(S_MM)
            D_MM[M1:M2, M1:M2] = S_MM[M1:M2, M1:M2]

            H_MM += iC_Mn @ (
                Co_nM @ D_MM @ Co_nM.T.conj() * homo +
                Cu_nM @ D_MM @ Cu_nM.T.conj() * lumo) @ iC_Mn.T.conj()

            a1 = a2
            M1 = M2

        return H_MM
