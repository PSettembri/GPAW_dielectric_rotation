import numpy as np

from gpaw.new.eigensolver import Eigensolver
from gpaw.new.lcao.hamiltonian import HamiltonianMatrixCalculator
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions


class LCAOEigensolver(Eigensolver):
    def __init__(self, basis):
        self.basis = basis

    def iterate(self, state, hamiltonian) -> float:
        matrix_calculator = hamiltonian.create_hamiltonian_matrix_calculator(
            state)

        for wfs in state.ibzwfs:
            self.iterate1(wfs, matrix_calculator)
        return 0.0

    def iterate1(self,
                 wfs: LCAOWaveFunctions,
                 matrix_calculator: HamiltonianMatrixCalculator):
        H_MM = matrix_calculator.calculate_matrix(wfs)
        eig_M = H_MM.eighg(wfs.L_MM, wfs.domain_comm)
        C_Mn = H_MM  # rename (H_MM now contains the eigenvectors)
        assert len(eig_M) >= wfs.nbands
        N = wfs.nbands
        wfs._eig_n = np.empty(wfs.nbands)
        wfs._eig_n[:] = eig_M[:N]
        comm = C_Mn.dist.comm
        if comm.size == 1:
            wfs.C_nM.data[:] = C_Mn.data.T[:N]
        else:
            C_Mn = C_Mn.gather(broadcast=True)
            n1, n2 = wfs.C_nM.dist.my_row_range()
            wfs.C_nM.data[:] = C_Mn.data.T[n1:n2]

        # Make sure wfs.C_nM and (lazy) wfs.P_ani are in sync:
        wfs._P_ani = None
