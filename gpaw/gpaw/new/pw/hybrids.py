from __future__ import annotations
from math import pi

import numpy as np
from gpaw.core import PWArray, PWDesc, UGDesc
from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.core.atom_arrays import AtomArrays
from gpaw.hybrids.paw import pawexxvv
from gpaw.hybrids.wstc import WignerSeitzTruncatedCoulomb
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pw.hamiltonian import PWHamiltonian
from gpaw.typing import Array1D
from gpaw.utilities import unpack_hermitian


def coulomb(pw: PWDesc, grid: UGDesc, omega: float):
    if omega == 0.0:
        wstc = WignerSeitzTruncatedCoulomb(
            pw.cell_cv, np.array([1, 1, 1]))
        return wstc.get_potential_new(pw, grid)

    v_G = pw.empty()
    G2_G = pw.ekin_G * 2
    v_G.data[:] = 4 * pi * (1 - np.exp(-G2_G / (4 * omega**2)))
    if pw.ng1 == 0:
        v_G.data[1:] /= G2_G[1:]
        v_G.data[0] = pi / omega**2
    else:
        v_G.data /= G2_G

    return v_G


class PWHybridHamiltonian(PWHamiltonian):
    def __init__(self,
                 grid: UGDesc,
                 pw: PWDesc,
                 xc,
                 setups,
                 fracpos_ac,
                 atomdist):
        super().__init__(grid, pw)
        self.grid = grid
        self.pw = pw
        self.exx_fraction = xc.exx_fraction
        self.exx_omega = xc.exx_omega

        self.exx_cc = sum(setup.ExxC for setup in setups) * self.exx_fraction
        self.VC_aii = [unpack_hermitian(setup.X_p * self.exx_fraction)
                       for setup in setups]
        self.delta_aiiL = [setup.Delta_iiL for setup in setups]
        self.C_app = [setup.M_pp * self.exx_fraction for setup in setups]

        self.v_G = coulomb(pw, grid, self.exx_omega)
        self.v_G.data *= self.exx_fraction
        self.ghat_aLG = setups.create_compensation_charges(
            pw, fracpos_ac, atomdist)
        self.plan = grid.fft_plans()

    def apply_orbital_dependent(self,
                                ibzwfs: IBZWaveFunctions,
                                D_asii,
                                psit2_nG: XArray,
                                spin: int,
                                Htpsit2_nG: XArray) -> None:
        assert isinstance(psit2_nG, PWArray)
        assert isinstance(Htpsit2_nG, PWArray)
        wfs = ibzwfs.wfs_qs[0][spin]
        D_aii = D_asii[:, spin]
        if ibzwfs.nspins == 1:
            D_aii.data *= 0.5
        f1_n = wfs.myocc_n
        psit1_nG = wfs.psit_nX
        P1_ani = wfs.P_ani
        pt_aiG = wfs.pt_aiX

        same = psit1_nG is psit2_nG

        if same:
            P2_ani = P1_ani
        else:
            P2_ani = pt_aiG.integrate(psit2_nG)

        evv, evc, ekin = self.calculate(D_aii, same, pt_aiG,
                                        psit1_nG, P1_ani, f1_n,
                                        psit2_nG, P2_ani,
                                        Htpsit2_nG)
        if same:
            for name, e in [('exx_vv', evv),
                            ('exx_vc', evc),
                            ('exx_kinetic', ekin)]:
                e *= ibzwfs.spin_degeneracy
                if spin == 0:
                    ibzwfs.energies[name] = e
                else:
                    ibzwfs.energies[name] += e
            ibzwfs.energies['exx_cc'] = self.exx_cc

    def calculate(self,
                  D_aii,
                  same,
                  pt_aiG,
                  psit1_nG: PWArray,
                  P1_ani: AtomArrays,
                  f1_n: Array1D,
                  psit2_nG: PWArray,
                  P2_ani: AtomArrays,
                  Htpsit2_nG: PWArray):

        psit1_R = self.grid.empty()
        psit2_R = self.grid.empty()
        rhot_R = self.grid.empty()
        rhot_G = self.pw.empty()
        vrhot_G = self.pw.empty()

        evv = 0.0
        evc = 0.0
        ekin = 0.0
        B_ani = {}
        for a, D_ii in D_aii.items():
            VV_ii = pawexxvv(self.C_app[a], D_ii)
            VC_ii = self.VC_aii[a]
            B_ni = P2_ani[a] @ (-VC_ii - 2 * VV_ii)
            B_ani[a] = B_ni
            if same:
                ec = (D_ii * VC_ii).sum()
                ev = (D_ii * VV_ii).sum()
                ekin += ec + 2 * ev
                evv -= ev
                evc -= ec

        for n2, (psit2_G, out_G) in enumerate(zip(psit2_nG, Htpsit2_nG)):
            psit2_G.ifft(out=psit2_R, plan=self.plan)

            for n1, (psit1_G, f1) in enumerate(zip(psit1_nG, f1_n)):
                psit1_G.ifft(out=psit1_R, plan=self.plan)
                rhot_R.data[:] = psit1_R.data * psit2_R.data
                rhot_R.fft(out=rhot_G, plan=self.plan)
                Q_aL = {a: np.einsum('i, ijL, j -> L',
                                     P1_ani[a][n1], delta_iiL, P2_ani[a][n2])
                        for a, delta_iiL in enumerate(self.delta_aiiL)}
                self.ghat_aLG.add_to(rhot_G, Q_aL)
                vrhot_G.data[:] = rhot_G.data * self.v_G.data
                if same:
                    e = f1 * f1_n[n2] * rhot_G.integrate(vrhot_G)
                    evv -= 0.5 * e
                    ekin += e
                vrhot_G.ifft(out=rhot_R, plan=self.plan)
                rhot_R.data *= psit1_R.data
                rhot_R.fft(out=rhot_G, plan=self.plan)
                out_G.data -= rhot_G.data * f1

                A_aL = self.ghat_aLG.integrate(vrhot_G)
                for a, A_L in A_aL.items():
                    B_ani[a][n2] -= np.einsum(
                        'L, ijL, j -> i',
                        f1 * A_L, self.delta_aiiL[a], P1_ani[a][n1])

        pt_aiG.add_to(Htpsit2_nG, B_ani)

        return evv, evc, ekin
