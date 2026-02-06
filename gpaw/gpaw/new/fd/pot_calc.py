from math import pi

import numpy as np

from gpaw.core import UGDesc
from gpaw.new import zips, spinsum, trace
from gpaw.new.pot_calc import PotentialCalculator


class FDPotentialCalculator(PotentialCalculator):
    def __init__(self,
                 wf_grid: UGDesc,
                 fine_grid: UGDesc,
                 setups,
                 xc,
                 poisson_solver,
                 *,
                 fracpos_ac,
                 atomdist,
                 interpolation_stencil_range=3,
                 xp=np):
        self.fine_grid = fine_grid
        self.grid = wf_grid

        self.vbar_ar = setups.create_local_potentials(fine_grid, fracpos_ac,
                                                      atomdist, xp=xp)
        self.ghat_aLr = setups.create_compensation_charges(fine_grid,
                                                           fracpos_ac,
                                                           atomdist,
                                                           xp=xp)

        self.vbar_r = fine_grid.empty(xp=xp)
        self.vbar_ar.to_uniform_grid(out=self.vbar_r)

        n = interpolation_stencil_range
        self.interpolation_stencil_range = n
        self._interpolate = wf_grid.transformer(fine_grid, n, xp=xp)
        self._restrict = fine_grid.transformer(wf_grid, n, xp=xp)

        self.xp = xp

        super().__init__(xc, poisson_solver, setups,
                         fracpos_ac=fracpos_ac)

    def __str__(self):
        txt = super().__str__()
        degree = self.interpolation_stencil_range * 2 - 1
        name = ['linear', 'cubic', 'quintic', 'heptic'][degree // 2]
        txt += (f'interpolation: tri-{name}' +
                f' # {degree}. degree polynomial\n')
        return txt

    def interpolate(self, a_xR, a_xr=None):
        return self._interpolate(a_xR, a_xr)

    def restrict(self, a_xr, a_xR=None):
        return self._restrict(a_xr, a_xR)

    def calculate_charges(self, vHt_r):
        return self.ghat_aLr.integrate(vHt_r)

    def calculate_non_selfconsistent_exc(self, xc, nt_sR, taut_sR):
        nt_sr, _, _ = self._interpolate_density(nt_sR)
        if taut_sR is not None:
            taut_sr = self.interpolate(taut_sR)
        else:
            taut_sr = None
        e_xc, _, _ = xc.calculate(nt_sr, taut_sr)
        return e_xc

    def _interpolate_density(self, nt_sR):
        nt_sr = self.interpolate(nt_sR)
        if not nt_sR.desc.pbc_c.all():
            Nt1_s = nt_sR.integrate()
            Nt2_s = nt_sr.integrate()
            for Nt1, Nt2, nt_r in zips(Nt1_s, Nt2_s, nt_sr):
                if float(Nt2) > 1e-14:
                    nt_r.data *= Nt1 / Nt2
        return nt_sr, None, None

    @trace
    def calculate_pseudo_potential(self, density, ibzwfs, vHt_r):
        nt_sr, _, _ = self._interpolate_density(density.nt_sR)
        grid2 = nt_sr.desc

        if density.taut_sR is not None:
            taut_sr = self.interpolate(density.taut_sR)
        else:
            taut_sr = None

        e_xc, vxct_sr, dedtaut_sr = self.xc.calculate(nt_sr, taut_sr)

        charge_r = grid2.empty(xp=self.xp)
        charge_r.data[:] = nt_sr.data[:density.ndensities].sum(axis=0)
        e_zero = self.vbar_r.integrate(charge_r)

        ccc_aL = density.calculate_compensation_charge_coefficients()

        # Normalize: (LCAO basis functions may extend outside box)
        comp_charge = (4 * pi)**0.5 * sum(float(ccc_L[0])
                                          for ccc_L in ccc_aL.values())
        comp_charge = ccc_aL.layout.atomdist.comm.sum_scalar(comp_charge)
        pseudo_charge = charge_r.integrate()
        charge_r.data *= -(comp_charge + density.charge) / pseudo_charge

        self.ghat_aLr.add_to(charge_r, ccc_aL)

        if vHt_r is None:
            vHt_r = grid2.zeros(xp=self.xp)
        self.poisson_solver.solve(vHt_r, charge_r)
        e_coulomb = 0.5 * vHt_r.integrate(charge_r)

        vt_sr = vxct_sr
        vt_sr.data += vHt_r.data + self.vbar_r.data
        vt_sR = self.restrict(vt_sr)

        e_external = 0.0

        return {'coulomb': e_coulomb,
                'zero': e_zero,
                'xc': e_xc,
                'external': e_external}, vt_sR, dedtaut_sr, vHt_r

    def move(self, fracpos_ac, atomdist):
        self.ghat_aLr.move(fracpos_ac, atomdist)
        self.vbar_ar.move(fracpos_ac, atomdist)
        self.vbar_ar.to_uniform_grid(out=self.vbar_r)

    def force_contributions(self, state):
        density = state.density
        potential = state.potential
        nt_R = spinsum(density.nt_sR)
        vt_R = spinsum(potential.vt_sR, mean=True)
        dedtaut_sR = potential.dedtaut_sR
        if dedtaut_sR is not None:
            dedtaut_R = spinsum(dedtaut_sR, mean=True)
            Ftauct_av = state.density.tauct_aX.derivative(dedtaut_R)
        else:
            Ftauct_av = None

        nt_r = self.interpolate(nt_R)
        if not nt_r.desc.pbc_c.all():
            scale = nt_R.integrate() / nt_r.integrate()
            nt_r.data *= scale

        return (self.ghat_aLr.derivative(state.potential.vHt_x),
                state.density.nct_aX.derivative(vt_R),
                Ftauct_av,
                self.vbar_ar.derivative(nt_r))
