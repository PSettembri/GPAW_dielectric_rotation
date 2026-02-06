import numpy as np
from functools import cached_property
from gpaw.response.pw_parallelization import Blocks1D
from gpaw.response.gamma_int import GammaIntegrator


class DielectricFunctionCalculator:
    def __init__(self, chi0, coulomb, xckernel, mode):
        self.coulomb = coulomb
        self.qpd = chi0.qpd
        self.mode = mode
        self.optical_limit = chi0.optical_limit
        self.chi0 = chi0
        self.xckernel = xckernel
        wblocks1d = Blocks1D(chi0.body.blockdist.blockcomm, len(chi0.wd))
        self.wblocks1d = wblocks1d
        # Generate fine grid in vicinity of gamma
        if chi0.optical_limit and wblocks1d.nlocal:
            self.gamma_int = GammaIntegrator(
                truncation=self.coulomb.truncation,
                kd=coulomb.kd, qpd=self.qpd,
                chi0_wvv=chi0.chi0_Wvv[wblocks1d.myslice],
                chi0_wxvG=chi0.chi0_WxvG[wblocks1d.myslice])
        else:
            self.gamma_int = None

    @cached_property
    def sqrtV_G(self):
        return self.coulomb.sqrtV(qpd=self.qpd, q_v=None)

    @cached_property
    def I_GG(self):
        return np.eye(self.qpd.ngmax)

    @cached_property
    def fxc_GG(self):
        if self.mode == 'GW':
            return self.I_GG
        else:
            return self.xckernel.calculate(self.qpd)

    def get_epsinv_wGG(self, only_correlation=True):
        """
        Calculates inverse dielectric matrix for all frequencies.
        """
        chi0_wGG = self.chi0.body.copy_array_with_distribution('wGG')
        epsinv_wGG = []
        for iw, chi0_GG in enumerate(chi0_wGG):
            epsinv_GG = self.get_epsinv_GG(chi0_GG, iw)
            if only_correlation:
                epsinv_GG -= self.I_GG
            epsinv_wGG.append(epsinv_GG)
        return np.asarray(epsinv_wGG)

    def get_epsinv_GG(self, chi0_GG, iw=None):
        """
        Calculates inverse dielectric matrix for single frequency
        """
        if self.optical_limit:
            return self._get_epsinvGamma_GG(chi0_GG, iw)
        dfc = _DielectricFunctionCalculator(self.sqrtV_G,
                                            chi0_GG,
                                            self.mode,
                                            self.fxc_GG)
        return dfc.get_epsinv_GG()

    def _get_epsinvGamma_GG(self, chi0_GG, iw):
        dfc = _DielectricFunctionCalculator(self.sqrtV_G,
                                            chi0_GG,
                                            self.mode,
                                            self.fxc_GG)
        return dfc.get_epsinvGamma_GG(self.gamma_int,
                                      self.qpd, iw,
                                      self.coulomb)


class _DielectricFunctionCalculator:
    def __init__(self, sqrtV_G, chi0_GG, mode, fxc_GG=None):
        self.sqrtV_G = sqrtV_G
        self.chiVV_GG = chi0_GG * sqrtV_G * sqrtV_G[:, np.newaxis]

        self.I_GG = np.eye(len(sqrtV_G))

        self.fxc_GG = fxc_GG
        self.chi0_GG = chi0_GG
        self.mode = mode

    def _chiVVfxc_GG(self):
        assert self.mode != 'GW'
        assert self.fxc_GG is not None
        return self.chiVV_GG @ self.fxc_GG

    def eps_GG_gwp(self):
        gwp_inv_GG = np.linalg.inv(self.I_GG - self._chiVVfxc_GG() +
                                   self.chiVV_GG)
        return self.I_GG - gwp_inv_GG @ self.chiVV_GG

    def eps_GG_gws(self):
        # Note how the signs are different wrt. gwp.
        # Nobody knows why.
        gws_inv_GG = np.linalg.inv(self.I_GG + self._chiVVfxc_GG() -
                                   self.chiVV_GG)
        return gws_inv_GG @ (self.I_GG - self.chiVV_GG)

    def eps_GG_plain(self):
        return self.I_GG - self.chiVV_GG

    def eps_GG_w_fxc(self):
        return self.I_GG - self._chiVVfxc_GG()

    def get_eps_GG(self):
        mode = self.mode
        if mode == 'GWP':
            return self.eps_GG_gwp()
        elif mode == 'GWS':
            return self.eps_GG_gws()
        elif mode == 'GW':
            return self.eps_GG_plain()
        elif mode == 'GWG':
            return self.eps_GG_w_fxc()
        raise ValueError(f'Unknown mode: {mode}')

    def get_epsinv_GG(self):
        eps_GG = self.get_eps_GG()
        return np.linalg.inv(eps_GG)

    def get_epsinvGamma_GG(self, gamma_int, qpd, iw, coulomb):
        # Get average epsinv over small region around Gamma
        epsinv_GG = np.zeros(self.chi0_GG.shape, complex)
        for iqf in range(len(gamma_int.qf_qv)):
            # Note! set_appendages changes (0,0), (0,:), (:,0)
            # elements of chi0_GG
            gamma_int.set_appendages(self.chi0_GG, iw, iqf)
            sqrtV_G = coulomb.sqrtV(
                qpd=qpd, q_v=gamma_int.qf_qv[iqf])
            dfc = _DielectricFunctionCalculator(
                sqrtV_G, self.chi0_GG, mode=self.mode, fxc_GG=self.fxc_GG)
            epsinv_GG += dfc.get_epsinv_GG() * gamma_int.weight_q
        return epsinv_GG
