from typing import TYPE_CHECKING

from gpaw.typing import Array1D, ArrayND
from gpaw.gpu import cupy as cp
import gpaw.cgpaw as cgpaw

__all__ = ['GPU_AWARE_MPI']

GPU_AWARE_MPI = getattr(cgpaw, 'gpu_aware_mpi', False)
GPU_ENABLED = getattr(cgpaw, 'GPU_ENABLED', False)


def add_to_density(f: float,
                   psit_X: ArrayND,
                   nt_X: ArrayND) -> None:
    nt_X += f * abs(psit_X)**2


def pw_precond(G2_G: Array1D,
               r_G: Array1D,
               ekin: float,
               o_G: Array1D) -> None:
    x = 1 / ekin / 3 * G2_G
    a = 27.0 + x * (18.0 + x * (12.0 + x * 8.0))
    xx = x * x
    o_G[:] = -4.0 / 3 / ekin * a / (a + 16.0 * xx * xx) * r_G


def pw_insert(coef_G: Array1D,
              Q_G: Array1D,
              x: float,
              array_Q: Array1D) -> None:
    array_Q[:] = 0.0
    array_Q.ravel()[Q_G] = x * coef_G


def pw_insert_gpu(psit_nG,
                  Q_G,
                  scale,
                  psit_bQ):
    assert scale == 1.0
    psit_bQ[:, Q_G] = psit_nG


def pwlfc_expand(f_Gs, emiGR_Ga, Y_GL,
                 l_s, a_J, s_J,
                 cc, f_GI):
    """
    f_GI = xp.empty((G2 - G1, self.nI), complex)
    I1 = 0
    for J, (a, s) in enumerate(zip(self.a_J, self.s_J)):
        l = self.l_s[s]
        I2 = I1 + 2 * l + 1
        f_GI[:, I1:I2] = (f_Gs[:, s] *
                          emiGR_Ga[:, a] *
                          Y_GL[:, l**2:(l + 1)**2].T *
                          (-1.0j)**l).T
        I1 = I2
    if cc:
        f_GI = f_GI.conj()
    if self.dtype == float:
        f_GI = f_GI.T.copy().view(float).T.copy()

    return f_GI
    """
    raise NotImplementedError


def pwlfc_expand_gpu(f_Gs, emiGR_Ga, Y_GL,
                     l_s, a_J, s_J,
                     cc, f_GI, I_J):
    raise NotImplementedError


def dH_aii_times_P_ani_gpu(dH_aii, ni_a,
                           P_nI, out_nI):
    I1 = 0
    J1 = 0
    for ni in ni_a._data:
        I2 = I1 + ni
        J2 = J1 + ni**2
        dH_ii = dH_aii[J1:J2].reshape((ni, ni))
        out_nI[:, I1:I2] = P_nI[:, I1:I2] @ dH_ii
        I1 = I2
        J1 = J2


def calculate_residuals_gpu(residual_nG, eps_n, wfs_nG):
    for residual_G, eps, wfs_G in zip(residual_nG, eps_n, wfs_nG):
        residual_G -= eps * wfs_G


def add_to_density_gpu(weight_n, psit_nR, nt_R):
    for weight, psit_R in zip(weight_n, psit_nR):
        nt_R += float(weight) * cp.abs(psit_R)**2


def symmetrize_ft(a_R, b_R, r_cc, t_c, offset_c):
    raise NotImplementedError


def evaluate_lda_gpu(nt_sr, vxct_sr, e_r) -> None:
    from gpaw.xc.kernel import XCKernel
    XCKernel('LDA').calculate(e_r._data, nt_sr._data, vxct_sr._data)


def evaluate_pbe_gpu(nt_sr, vxct_sr, e_r, sigma_xr, dedsigma_xr) -> None:
    from gpaw.xc.kernel import XCKernel
    XCKernel('PBE').calculate(e_r._data, nt_sr._data, vxct_sr._data,
                              sigma_xr._data, dedsigma_xr._data)


if not TYPE_CHECKING:
    from gpaw.cgpaw import (  # noqa
        add_to_density, pw_precond, pw_insert,
        pwlfc_expand, symmetrize_ft)

    if GPU_ENABLED:
        from gpaw.cgpaw import (  # noqa
            pwlfc_expand_gpu, add_to_density_gpu, pw_insert_gpu,
            dH_aii_times_P_ani_gpu, evaluate_lda_gpu, evaluate_pbe_gpu,
            calculate_residuals_gpu)
