from __future__ import annotations

import numpy as np
from gpaw.core.atom_arrays import (AtomArrays, AtomArraysLayout,
                                   AtomDistribution)
from gpaw.core.matrix import Matrix
from gpaw.mpi import MPIComm, receive, send, serial_comm
from gpaw.new.potential import Potential
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.new.wave_functions import WaveFunctions
from gpaw.setup import Setups
from gpaw.typing import Array2D


class LCAOWaveFunctions(WaveFunctions):
    xp = np

    def __init__(self,
                 *,
                 setups: Setups,
                 tci_derivatives,
                 basis,
                 C_nM: Matrix,
                 S_MM: Matrix,
                 T_MM: Matrix,
                 P_aMi,
                 fracpos_ac: Array2D,
                 atomdist: AtomDistribution,
                 kpt_c=(0.0, 0.0, 0.0),
                 domain_comm: MPIComm = serial_comm,
                 spin: int = 0,
                 q: int = 0,
                 k: int = 0,
                 weight: float = 1.0,
                 ncomponents: int = 1):
        super().__init__(setups=setups,
                         nbands=C_nM.shape[0],
                         spin=spin,
                         q=q,
                         k=k,
                         kpt_c=kpt_c,
                         weight=weight,
                         fracpos_ac=fracpos_ac,
                         atomdist=atomdist,
                         ncomponents=ncomponents,
                         dtype=C_nM.dtype,
                         domain_comm=domain_comm,
                         band_comm=C_nM.dist.comm)
        self.tci_derivatives = tci_derivatives
        self.basis = basis
        self.C_nM = C_nM
        self.T_MM = T_MM
        self.S_MM = S_MM
        self.P_aMi = P_aMi

        self.bytes_per_band = (self.array_shape(global_shape=True)[0] *
                               C_nM.data.itemsize)

        # This is for TB-mode (and MYPY):
        self.V_MM: Matrix

        self._L_MM = None

    def move(self,
             fracpos_ac: Array2D,
             atomdist: AtomDistribution) -> None:
        super().move(fracpos_ac, atomdist)
        self._L_MM = None

    @property
    def L_MM(self):
        if self._L_MM is None:
            S_MM = self.S_MM.copy()
            S_MM.invcholesky()
            if self.ncomponents < 4:
                self._L_MM = S_MM
            else:
                M, M = S_MM.shape
                L_sMsM = Matrix(2 * M, 2 * M, dtype=complex)
                L_sMsM.data[:] = 0.0
                L_sMsM.data[:M, :M] = S_MM.data
                L_sMsM.data[M:, M:] = S_MM.data
                self._L_MM = L_sMsM
        return self._L_MM

    def _short_string(self, global_shape):
        return f'basis functions: {global_shape[0]}'

    def array_shape(self, global_shape=False):
        if global_shape:
            return self.C_nM.shape[1:]
        1 / 0

    @property
    def P_ani(self):
        if self._P_ani is None:
            atomdist = AtomDistribution.from_atom_indices(
                list(self.P_aMi),
                self.domain_comm,
                natoms=len(self.setups))
            layout = AtomArraysLayout([setup.ni for setup in self.setups],
                                      atomdist=atomdist,
                                      dtype=self.dtype)
            self._P_ani = layout.empty(self.nbands,
                                       comm=self.C_nM.dist.comm)
            for a, P_Mi in self.P_aMi.items():
                self._P_ani[a][:] = self.C_nM.data @ P_Mi

        return self._P_ani

    def add_to_density(self,
                       nt_sR,
                       D_asii: AtomArrays) -> None:
        """Add density from wave functions.

        Adds to ``nt_sR`` and ``D_asii``.
        """
        rho_MM = self.calculate_density_matrix()
        self.basis.construct_density(rho_MM, nt_sR.data[self.spin], q=self.q)
        f_n = self.weight * self.spin_degeneracy * self.myocc_n
        self.add_to_atomic_density_matrices(f_n, D_asii)

    def gather_wave_function_coefficients(self) -> np.ndarray:
        C_nM = self.C_nM.gather()
        if C_nM is not None:
            return C_nM.data
        return None

    def calculate_density_matrix(self, eigs=False) -> np.ndarray:
        """Calculate the density matrix.

        The density matrix is:::

                -- *
          ρ   = > C  C   f
           μν   -- nμ nν  n
                n

        Returns
        -------
        The density matrix in the LCAO basis
        """
        if self.domain_comm.rank == 0:
            f_n = self.weight * self.spin_degeneracy * self.myocc_n
            if eigs:
                f_n *= self.myeig_n
            C_nM = self.C_nM.data
            rho_MM = (C_nM.T.conj() * f_n) @ C_nM
            self.band_comm.sum(rho_MM)
        else:
            rho_MM = np.empty_like(self.T_MM.data)
        self.domain_comm.broadcast(rho_MM, 0)

        return rho_MM

    def to_uniform_grid_wave_functions(self,
                                       grid,
                                       basis):
        grid = grid.new(kpt=self.kpt_c, dtype=self.dtype)
        psit_nR = grid.zeros(self.nbands, self.band_comm)
        basis.lcao_to_grid(self.C_nM.data, psit_nR.data, self.q)

        return PWFDWaveFunctions.from_wfs(self, psit_nR)

    def collect(self,
                n1: int = 0,
                n2: int = 0) -> LCAOWaveFunctions | None:
        # Quick'n'dirty implementation
        # We should generalize the PW+FD method
        assert self.band_comm.size == 1
        n2 = n2 or self.nbands + n2
        return LCAOWaveFunctions(
            setups=self.setups,
            tci_derivatives=self.tci_derivatives,
            basis=self.basis,
            C_nM=Matrix(n2 - n1,
                        self.C_nM.shape[1],
                        data=self.C_nM.data[n1:n2].copy()),
            S_MM=self.S_MM,
            T_MM=self.T_MM,
            P_aMi=self.P_aMi,
            fracpos_ac=self.fracpos_ac,
            atomdist=self.atomdist.gather(),
            kpt_c=self.kpt_c,
            spin=self.spin,
            q=self.q,
            k=self.k,
            weight=self.weight,
            ncomponents=self.ncomponents)

    def force_contribution(self, potential: Potential, F_av: Array2D):
        from gpaw.new.lcao.forces import add_force_contributions
        add_force_contributions(self, potential, F_av)
        return F_av

    def send(self, rank, comm):
        stuff = (self.kpt_c,
                 self.C_nM.data,
                 self.spin,
                 self.q,
                 self.k,
                 self.weight,
                 self.ncomponents)
        send(stuff, rank, comm)

    def receive(self, rank, comm):
        kpt_c, data, spin, q, k, weight, ncomponents = receive(rank, comm)
        return LCAOWaveFunctions(setups=self.setups,
                                 tci_derivatives=self.tci_derivatives,
                                 basis=self.basis,
                                 C_nM=Matrix(*data.shape, data=data),
                                 S_MM=None,
                                 T_MM=None,
                                 P_aMi=None,
                                 fracpos_ac=self.fracpos_ac,
                                 atomdist=self.atomdist.gather(),
                                 kpt_c=kpt_c,
                                 spin=spin,
                                 q=q,
                                 k=k,
                                 weight=weight,
                                 ncomponents=ncomponents)
