import numpy as np
from gpaw.core import PWDesc
from gpaw.gpu import cupy as cp
from gpaw.mpi import broadcast_float
from gpaw.new import zips, spinsum
from gpaw.new.pot_calc import PotentialCalculator
from gpaw.new.pw.stress import calculate_stress
from gpaw.setup import Setups
from gpaw.new.calculation import DFTState


class PlaneWavePotentialCalculator(PotentialCalculator):
    def __init__(self,
                 grid,
                 fine_grid,
                 pw: PWDesc,
                 fine_pw: PWDesc,
                 setups: Setups,
                 xc,
                 poisson_solver,
                 *,
                 external_potential,
                 fracpos_ac,
                 atomdist,
                 soc=False,
                 xp=np):
        self.xp = xp
        super().__init__(xc, poisson_solver, setups,
                         external_potential=external_potential,
                         fracpos_ac=fracpos_ac,
                         soc=soc)

        self.vbar_ag = setups.create_local_potentials(
            pw, fracpos_ac, atomdist, xp)
        self.ghat_aLh = setups.create_compensation_charges(
            fine_pw, fracpos_ac, atomdist, xp)

        self.pw = pw
        self.fine_pw = fine_pw
        self.pw0 = pw.new(comm=None)  # not distributed

        self.h_g, self.g_r = fine_pw.map_indices(self.pw0)
        if xp is cp:
            self.h_g = cp.asarray(self.h_g)
            self.g_r = [cp.asarray(g) for g in self.g_r]

        self.fftplan = grid.fft_plans(xp=xp)
        self.fftplan2 = fine_grid.fft_plans(xp=xp)

        self.grid = grid
        self.fine_grid = fine_grid

        self.vbar_g = pw.zeros(xp=xp)
        self.vbar_ag.add_to(self.vbar_g)
        self.vbar0_g = self.vbar_g.gather()

        # For forces and stress:
        self._nt_g = None
        self._vt_g = None
        self._dedtaut_g = None

    def interpolate(self, a_R, a_r=None):
        return a_R.interpolate(self.fftplan, self.fftplan2,
                               grid=self.fine_grid, out=a_r)

    def restrict(self, a_r, a_R=None):
        return a_r.fft_restrict(self.fftplan2, self.fftplan,
                                grid=self.grid, out=a_R)

    def calculate_charges(self, vHt_h):
        return self.ghat_aLh.integrate(vHt_h)

    def _interpolate_density(self, nt_sR):
        nt_sr = self.fine_grid.empty(nt_sR.dims, xp=self.xp)
        pw = self.vbar_g.desc

        if pw.comm.rank == 0:
            indices = self.xp.asarray(self.pw0.indices(self.fftplan.shape))
            nt0_g = self.pw0.zeros(xp=self.xp)
        else:
            nt0_g = None

        ndensities = nt_sR.dims[0] % 3
        for spin, (nt_R, nt_r) in enumerate(zips(nt_sR, nt_sr)):
            self.interpolate(nt_R, nt_r)
            if spin < ndensities and pw.comm.rank == 0:
                nt0_g.data += self.xp.asarray(
                    self.fftplan.tmp_Q.ravel()[indices])

        return nt_sr, pw, nt0_g

    def _interpolate_and_calculate_xc(self, xc, nt_sR, ibzwfs):
        ...

    def calculate_non_selfconsistent_exc(self, xc, nt_sR, ibzwfs):
        _, _, _, _, e_xc, _ = self._interpolate_and_calculate_xc(
            xc, nt_sR, ibzwfs)
        return e_xc

    def calculate_pseudo_potential(self, density, ibzwfs, vHt_h=None):
        nt_sr, pw, nt0_g = self._interpolate_density(density.nt_sR)

        if density.taut_sR is not None:
            taut_sr = self.interpolate(density.taut_sR)
        else:
            taut_sr = None

        e_xc, vxct_sr, dedtaut_sr = self.xc.calculate(nt_sr, taut_sr)

        if pw.comm.rank == 0:
            nt0_g.data *= 1 / np.prod(density.nt_sR.desc.size_c)
            e_zero = self.vbar0_g.integrate(nt0_g)
        else:
            e_zero = 0.0
        e_zero = broadcast_float(float(e_zero), pw.comm)

        if vHt_h is None:
            vHt_h = self.ghat_aLh.pw.zeros(xp=self.xp)

        charge_h = vHt_h.desc.zeros(xp=self.xp)
        coef_aL = density.calculate_compensation_charge_coefficients()
        self.ghat_aLh.add_to(charge_h, coef_aL)

        if pw.comm.rank == 0:
            for rank, g in enumerate(self.g_r):
                if rank == 0:
                    charge_h.data[self.h_g] += nt0_g.data[g]
                else:
                    pw.comm.send(nt0_g.data[g], rank)
        else:
            data = self.xp.empty(len(self.h_g), complex)
            pw.comm.receive(data, 0)
            charge_h.data[self.h_g] += data

        # background charge ???

        e_coulomb = self.poisson_solver.solve(vHt_h, charge_h)

        if pw.comm.rank == 0:
            vt0_g = self.vbar0_g.copy()
            for rank, g in enumerate(self.g_r):
                if rank == 0:
                    vt0_g.data[g] += vHt_h.data[self.h_g]
                else:
                    data = self.xp.empty(len(g), complex)
                    pw.comm.receive(data, rank)
                    vt0_g.data[g] += data
            vt0_R = vt0_g.ifft(
                plan=self.fftplan,
                grid=density.nt_sR.desc.new(comm=None))
        else:
            pw.comm.send(vHt_h.data[self.h_g], 0)

        vt_sR = density.nt_sR.new()
        vt_sR[0].scatter_from(vt0_R if pw.comm.rank == 0 else None)
        if density.ndensities == 2:
            vt_sR.data[1] = vt_sR.data[0]
        vt_sR.data[density.ndensities:] = 0.0

        e_external = self.external_potential.update_potential(vt_sR, density)

        vtmp_R = vt_sR.desc.empty(xp=self.xp)
        for spin, (vt_R, vxct_r) in enumerate(zips(vt_sR, vxct_sr)):
            self.restrict(vxct_r, vtmp_R)
            vt_R.data += vtmp_R.data

        self._reset()

        return {'coulomb': e_coulomb,
                'zero': e_zero,
                'xc': e_xc,
                'stress': e_coulomb + e_zero,
                'external': e_external}, vt_sR, dedtaut_sr, vHt_h

    def move(self, fracpos_ac, atomdist):
        self.ghat_aLh.move(fracpos_ac, atomdist)
        self.vbar_ag.move(fracpos_ac, atomdist)
        self.vbar_g.data[:] = 0.0
        self.vbar_ag.add_to(self.vbar_g)
        self.vbar0_g = self.vbar_g.gather()
        self._reset()

    def _reset(self):
        self._vt_g = None
        self._nt_g = None
        self._dedtaut_g = None

    def _force_stress_helper(self, state: DFTState):
        """Only do the work once - in case both forces and stress is needed"""
        if self._vt_g is not None:
            return self._vt_g, self._nt_g, self._dedtaut_g

        density = state.density
        potential = state.potential
        nt_R = spinsum(density.nt_sR)
        vt_R = spinsum(potential.vt_sR, mean=True)
        self._vt_g = vt_R.fft(self.fftplan, pw=self.pw)
        self._nt_g = nt_R.fft(self.fftplan, pw=self.pw)

        dedtaut_sR = potential.dedtaut_sR
        if dedtaut_sR is not None:
            dedtaut_R = spinsum(dedtaut_sR, mean=True)
            self._dedtaut_g = dedtaut_R.fft(self.fftplan, pw=self.pw)
        else:
            self._dedtaut_g = None

        return self._vt_g, self._nt_g, self._dedtaut_g

    def force_contributions(self, state):
        vt_g, nt_g, dedtaut_g = self._force_stress_helper(state)
        if dedtaut_g is None:
            Ftauct_av = None
        else:
            Ftauct_av = state.density.tauct_aX.derivative(dedtaut_g)

        return (self.ghat_aLh.derivative(state.potential.vHt_x),
                state.density.nct_aX.derivative(vt_g),
                Ftauct_av,
                self.vbar_ag.derivative(nt_g))

    def stress(self, state):
        vt_g, nt_g, dedtaut_g = self._force_stress_helper(state)
        return calculate_stress(self, state, vt_g, nt_g, dedtaut_g)
