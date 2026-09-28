import numpy as np
from ase.dft.kpoints import monkhorst_pack


class GammaIntegrator:
    def __init__(self, truncation, kd, qpd, chi0_wvv, chi0_wxvG):
        N = 4
        N_c = np.array([N, N, N])
        if truncation is not None:
            # Only average periodic directions if trunction is used
            N_c[kd.N_c == 1] = 1
        qf_qc = monkhorst_pack(N_c) / kd.N_c
        qf_qc *= 1.0e-6
        # XXX previously symmetry was used in Gamma integrator.
        # This was not correct, as explained in #709.
        self.weight_q = 1. / np.prod(N_c)
        self.qf_qv = 2 * np.pi * (qf_qc @ qpd.gd.icell_cv)
        self.a_wq = np.sum([chi0_vq * self.qf_qv.T
                            for chi0_vq in
                            np.dot(chi0_wvv, self.qf_qv.T)],
                           axis=1)
        self.a0_qwG = np.dot(self.qf_qv, chi0_wxvG[:, 0])
        self.a1_qwG = np.dot(self.qf_qv, chi0_wxvG[:, 1])

    def set_appendages(self, chi0_GG, iw, iqf):
        # Most likely this method should be moved to a Chi0Appendages class.
        chi0_GG[0, :] = self.a0_qwG[iqf, iw]
        chi0_GG[:, 0] = self.a1_qwG[iqf, iw]
        chi0_GG[0, 0] = self.a_wq[iw, iqf]
