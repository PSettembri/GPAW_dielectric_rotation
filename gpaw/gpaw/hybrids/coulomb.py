from math import pi

import numpy as np

from gpaw.hybrids.wstc import WignerSeitzTruncatedCoulomb as WSTC


def coulomb_interaction(omega, gd, kd):
    if omega:
        return ShortRangeCoulomb(omega)
    return WSTC(gd.cell_cv, kd.N_c)


class ShortRangeCoulomb:
    def __init__(self, omega):
        self.omega = omega

    def get_description(self):
        return f'Short-range Coulomb: erfc(omega*r)/r (omega = {self.omega} ' \
               'bohr^-1)'

    def get_potential(self, pd):
        G2_G = pd.G2_qG[0]
        x_G = 1 - np.exp(-G2_G / (4 * self.omega**2))
        with np.errstate(invalid='ignore'):
            v_G = 4 * pi * x_G / G2_G
        G0 = G2_G.argmin()
        if G2_G[G0] < 1e-11:
            v_G[G0] = pi / self.omega**2
        return v_G
