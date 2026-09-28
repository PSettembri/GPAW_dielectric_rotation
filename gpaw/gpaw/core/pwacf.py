from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

import numpy as np
from gpaw.core.atom_arrays import AtomArraysLayout, AtomDistribution
from gpaw.core.atom_centered_functions import AtomCenteredFunctions
from gpaw.core.uniform_grid import UGArray
from gpaw.ffbt import rescaled_fourier_bessel_transform
from gpaw.gpu import cupy_is_fake
from gpaw.lfc import BaseLFC
from gpaw.new import prod
from gpaw.new.c import pwlfc_expand, pwlfc_expand_gpu
from gpaw.spherical_harmonics import Y, nablarlYL
from gpaw.utilities.blas import mmm

if TYPE_CHECKING:
    from gpaw.core.plane_waves import PWDesc


class PWAtomCenteredFunctions(AtomCenteredFunctions):
    def __init__(self,
                 functions,
                 fracpos,
                 pw,
                 atomdist=None,
                 xp=None):
        AtomCenteredFunctions.__init__(self, functions, fracpos, atomdist)
        self.pw = pw
        self.xp = xp or np

    def new(self, pw, atomdist):
        return PWAtomCenteredFunctions(
            self.functions,
            self.fracpos_ac,
            pw,
            atomdist=atomdist,
            xp=self.xp)

    def _lazy_init(self):
        if self._lfc is not None:
            return

        self._lfc = PWLFC(self.functions, self.pw, xp=self.xp)
        if self._atomdist is None:
            self._atomdist = AtomDistribution.from_number_of_atoms(
                len(self.fracpos_ac), self.pw.comm)

        self._lfc.set_positions(self.fracpos_ac, self._atomdist)
        self._layout = AtomArraysLayout([sum(2 * f.l + 1 for f in funcs)
                                         for funcs in self.functions],
                                        self._atomdist,
                                        self.pw.dtype,
                                        xp=self.xp)

    def __repr__(self):
        s = super().__repr__()
        if self.xp is np:
            return s
        return s[:-1] + ', xp=cp)'

    def to_uniform_grid(self,
                        out: UGArray,
                        scale: float = 1.0) -> UGArray:
        out_G = self.pw.zeros(xp=out.xp)
        self.add_to(out_G, scale)
        return out_G.ifft(out=out)

    def change_cell(self, new_pw):
        self.pw = new_pw
        self._lfc = None


class PWLFC(BaseLFC):
    def __init__(self,
                 functions,
                 pw: PWDesc,
                 blocksize=5000, *, xp):
        """Reciprocal-space plane-wave localized function collection.

        spline_aj: list of list of spline objects
            Splines.
        pd: PWDescriptor
            Plane-wave descriptor object.
        blocksize: int
            Block-size to use when looping over G-vectors.  Use None for
            doing all G-vectors in one big block.
        """

        self.xp = xp
        self.pw = pw
        self.spline_aj = functions

        self.dtype = pw.dtype

        self.initialized = False

        # These will be filled in later:
        self.Y_GL = None
        self.emiGR_Ga = None
        self.f_Gs = None
        self.l_s = None
        self.a_J = None
        self.s_J = None
        self.lmax = None

        if blocksize is not None:
            if pw.maxmysize <= blocksize:
                # No need to block G-vectors
                blocksize = None
        self.blocksize = blocksize

        # These are set later in set_potitions():
        self.eikR_a = None
        self.my_atom_indices = None
        self.my_indices = None
        self.pos_av = None
        self.nI = None

        self.comm = pw.comm

    def initialize(self):
        """Initialize position-independent stuff."""
        if self.initialized:
            return

        xp = self.xp

        splines = {}  # Dict[Spline, int]
        for spline_j in self.spline_aj:
            for spline in spline_j:
                if spline not in splines:
                    splines[spline] = len(splines)
        nsplines = len(splines)

        nJ = sum(len(spline_j) for spline_j in self.spline_aj)

        self.f_Gs = xp.empty(self.pw.myshape + (nsplines,))
        self.l_s = np.empty(nsplines, np.int32)
        self.a_J = np.empty(nJ, np.int32)
        self.s_J = np.empty(nJ, np.int32)
        self.I_J = np.empty(nJ, np.int32)
        # Fourier transform radial functions:
        J = 0
        done = set()  # Set[Spline]
        I = 0
        for a, spline_j in enumerate(self.spline_aj):
            for spline in spline_j:
                s = splines[spline]  # get spline index
                if spline not in done:
                    f = rescaled_fourier_bessel_transform(spline)
                    G_G = (2 * self.pw.ekin_G)**0.5
                    self.f_Gs[:, s] = xp.asarray(f.map(G_G))
                    self.l_s[s] = spline.get_angular_momentum_number()
                    done.add(spline)
                self.a_J[J] = a
                self.s_J[J] = s
                self.I_J[J] = I
                I += 2 * spline.get_angular_momentum_number() + 1
                J += 1

        self.lmax = max(self.l_s, default=-1)

        # Spherical harmonics:
        G_Gv = self.pw.G_plus_k_Gv
        self.Y_GL = xp.empty((len(G_Gv), (self.lmax + 1)**2))
        for L in range((self.lmax + 1)**2):
            self.Y_GL[:, L] = xp.asarray(Y(L, *G_Gv.T))

        self.l_s = xp.asarray(self.l_s)
        self.a_J = xp.asarray(self.a_J)
        self.s_J = xp.asarray(self.s_J)
        self.I_J = xp.asarray(self.I_J)

        self.initialized = True

    def get_function_count(self, a):
        return sum(2 * spline.get_angular_momentum_number() + 1
                   for spline in self.spline_aj[a])

    def set_positions(self, spos_ac, atomdist):
        self.initialize()

        xp = self.xp

        if self.pw.dtype == float:
            self.eikR_a = xp.ones(len(spos_ac))
        else:
            self.eikR_a = xp.asarray(
                np.exp(2j * pi * (spos_ac @ self.pw.kpt_c)))

        self.pos_av = np.dot(spos_ac, self.pw.cell)

        Gk_Gv = xp.asarray(self.pw.G_plus_k_Gv)
        GkR_Ga = Gk_Gv @ xp.asarray(self.pos_av.T)
        self.emiGR_Ga = xp.exp(-1j * GkR_Ga) * self.eikR_a

        rank_a = atomdist.rank_a

        self.my_atom_indices = []
        self.my_indices = []
        I1 = 0
        for a, rank in enumerate(rank_a):
            I2 = I1 + self.get_function_count(a)
            if rank == self.comm.rank:
                self.my_atom_indices.append(a)
                self.my_indices.append((a, I1, I2))
            I1 = I2
        self.nI = I1

    def expand(self, G1=0, G2=None, cc=False):
        """Expand functions in plane-waves.

        q: int
            k-point index.
        G1: int
            Start G-vector index.
        G2: int
            End G-vector index.
        cc: bool
            Complex conjugate.
        """
        xp = self.xp

        if G2 is None:
            G2 = self.Y_GL.shape[0]

        emiGR_Ga = self.emiGR_Ga[G1:G2]
        f_Gs = self.f_Gs[G1:G2]
        Y_GL = self.Y_GL[G1:G2]

        if self.dtype == complex:
            f_GI = xp.empty((G2 - G1, self.nI), complex)
        else:
            # Special layout because BLAS does not have real-complex
            # multiplications.  f_GI(G,I) layout:
            #
            #    real(G1, 0),   real(G1, 1),   ...
            #    imag(G1, 0),   imag(G1, 1),   ...
            #    real(G1+1, 0), real(G1+1, 1), ...
            #    imag(G1+1, 0), imag(G1+1, 1), ...
            #    ...

            f_GI = xp.empty((2 * (G2 - G1), self.nI))

        if xp is np:
            # Fast C-code:
            pwlfc_expand(f_Gs, emiGR_Ga, Y_GL,
                         self.l_s, self.a_J, self.s_J,
                         cc, f_GI)
        elif cupy_is_fake:
            pwlfc_expand(f_Gs._data, emiGR_Ga._data, Y_GL._data,
                         self.l_s._data, self.a_J._data, self.s_J._data,
                         cc, f_GI._data)
        else:
            pwlfc_expand_gpu(f_Gs, emiGR_Ga, Y_GL,
                             self.l_s, self.a_J, self.s_J,
                             cc, f_GI, self.I_J)
        return f_GI

    def block(self, ensure_same_number_of_blocks=False):
        nG = self.Y_GL.shape[0]
        B = self.blocksize
        if B:
            G1 = 0
            while G1 < nG:
                G2 = min(G1 + B, nG)
                yield G1, G2
                G1 = G2
            if ensure_same_number_of_blocks:
                # Make sure we yield the same number of times:
                nb = (self.pw.maxmysize + B - 1) // B
                mynb = (nG + B - 1) // B
                if mynb < nb:
                    yield nG, nG  # empty block
        else:
            yield 0, nG

    def add(self, a_xG, c_axi=1.0, q=None):
        if self.nI == 0:
            return
        c_xI = self.xp.empty(a_xG.shape[:-1] + (self.nI,), self.dtype)

        if isinstance(c_axi, float):
            assert a_xG.ndim == 1
            c_xI[:] = c_axi
        else:
            if self.comm.size != 1:
                c_xI[:] = 0.0
            for a, I1, I2 in self.my_indices:
                c_xI[..., I1:I2] = c_axi[a] * self.eikR_a[a].conj()
            if self.comm.size != 1:
                self.comm.sum(c_xI)

        nx = prod(c_xI.shape[:-1])
        if nx == 0:
            return
        c_xI = c_xI.reshape((nx, self.nI))
        a_xG = a_xG.reshape((nx, a_xG.shape[-1])).view(self.dtype)

        for G1, G2 in self.block():
            f_GI = self.expand(G1, G2, cc=False)

            if self.dtype == float:
                # f_IG = f_IG.view(float)
                G1 *= 2
                G2 *= 2

            if self.xp is np:
                mmm(1.0 / self.pw.dv, c_xI, 'N', f_GI, 'T',
                    1.0, a_xG[:, G1:G2])
            else:
                self.xp.cublas.gemm('N', 'T',
                                    c_xI, f_GI, a_xG[:, G1:G2],
                                    1.0 / self.pw.dv, 1.0)

    def integrate(self, a_xG, c_axi=None, q=-1):
        xp = self.xp
        if self.nI == 0:
            return c_axi
        c_xI = xp.zeros(a_xG.shape[:-1] + (self.nI,), self.dtype)

        nx = prod(c_xI.shape[:-1])
        if nx == 0:
            return
        b_xI = c_xI.reshape((nx, self.nI))
        a_xG = a_xG.reshape((nx, a_xG.shape[-1]))

        alpha = 1.0  # / self.pd.gd.N_c.prod()
        if self.dtype == float:
            alpha *= 2
            a_xG = a_xG.view(float)

        if c_axi is None:
            c_axi = self.dict(a_xG.shape[:-1])

        x = 0.0
        for G1, G2 in self.block():
            f_GI = self.expand(G1, G2, cc=self.dtype == complex)
            if self.dtype == float:
                if G1 == 0 and self.comm.rank == 0:
                    f_GI[0] *= 0.5
                G1 *= 2
                G2 *= 2
            if xp is np:
                mmm(alpha, a_xG[:, G1:G2], 'N', f_GI, 'N', x, b_xI)
            else:
                xp.cublas.gemm('N', 'N',
                               a_xG[:, G1:G2], f_GI, b_xI,
                               alpha, x)
            x = 1.0

        self.comm.sum(b_xI)
        for a, I1, I2 in self.my_indices:
            c_axi[a][:] = self.eikR_a[a] * c_xI[..., I1:I2]

        return c_axi

    def derivative(self, a_xG, c_axiv=None, q=-1):
        xp = self.xp
        c_vxI = xp.zeros((3,) + a_xG.shape[:-1] + (self.nI,), self.dtype)
        nx = prod(c_vxI.shape[1:-1])
        if nx == 0:
            return
        b_vxI = c_vxI.reshape((3, nx, self.nI))
        a_xG = a_xG.reshape((nx, a_xG.shape[-1])).view(self.dtype)

        alpha = 1.0

        if c_axiv is None:
            c_axiv = self.dict(a_xG.shape[:-1], derivative=True)

        x = 0.0
        for G1, G2 in self.block():
            f_GI = self.expand(G1, G2, cc=True)
            G_Gv = xp.asarray(self.pw.G_plus_k_Gv[G1:G2])
            if self.dtype == float:
                d_GI = xp.empty(f_GI.shape)
                for v in range(3):
                    d_GI[::2] = f_GI[1::2] * G_Gv[:, v, np.newaxis]
                    d_GI[1::2] = f_GI[::2] * G_Gv[:, v, np.newaxis]
                    if xp is np:
                        mmm(2 * alpha,
                            a_xG[:, 2 * G1:2 * G2], 'N',
                            d_GI, 'N',
                            x, b_vxI[v])
                    else:
                        xp.cublas.gemm('N', 'N',
                                       a_xG[:, 2 * G1:2 * G2],
                                       d_GI,
                                       b_vxI[v],
                                       2 * alpha, x)
            else:
                for v in range(3):
                    if xp is np:
                        mmm(-alpha,
                            a_xG[:, G1:G2], 'N',
                            f_GI * G_Gv[:, v, np.newaxis], 'N',
                            x, b_vxI[v])
                    else:
                        xp.cublas.gemm('N', 'N',
                                       a_xG[:, G1:G2],
                                       f_GI * G_Gv[:, v, np.newaxis],
                                       b_vxI[v],
                                       -alpha, x)
            x = 1.0

        self.comm.sum(c_vxI)

        for v in range(3):
            if self.dtype == float:
                for a, I1, I2 in self.my_indices:
                    c_axiv[a][..., v] = c_vxI[v, ..., I1:I2]
            else:
                for a, I1, I2 in self.my_indices:
                    c_axiv[a][..., v] = (1.0j * self.eikR_a[a] *
                                         c_vxI[v, ..., I1:I2])

        return c_axiv

    def stress_tensor_contribution(self, a_xG, c_axi=1.0):
        xp = self.xp
        cache = {}
        things = []
        I1 = 0
        lmax = 0
        for a, spline_j in enumerate(self.spline_aj):
            for spline in spline_j:
                if spline not in cache:
                    s = rescaled_fourier_bessel_transform(spline)
                    G_G = (2 * self.pw.ekin_G)**0.5
                    f_G = []
                    dfdGoG_G = []
                    for G in G_G:
                        f, dfdG = s.get_value_and_derivative(G)
                        if G < 1e-10:
                            G = 1.0
                        f_G.append(f)
                        dfdGoG_G.append(dfdG / G)
                    f_G = xp.array(f_G)
                    dfdGoG_G = xp.array(dfdGoG_G)
                    cache[spline] = (f_G, dfdGoG_G)
                else:
                    f_G, dfdGoG_G = cache[spline]
                l = spline.l
                lmax = max(l, lmax)
                I2 = I1 + 2 * l + 1
                things.append((a, l, I1, I2, f_G, dfdGoG_G))
                I1 = I2

        if isinstance(c_axi, float):
            c_axi = {a: c_axi for a in range(len(self.pos_av))}

        G0_Gv = self.pw.G_plus_k_Gv

        stress_vv = xp.zeros((3, 3))
        for G1, G2 in self.block(ensure_same_number_of_blocks=True):
            G_Gv = G0_Gv[G1:G2]
            Z_LvG = xp.array([nablarlYL(L, G_Gv.T)
                              for L in range((lmax + 1)**2)])
            G_Gv = xp.asarray(G_Gv)
            aa_xG = a_xG[..., G1:G2]
            for v1 in range(3):
                for v2 in range(3):
                    stress_vv[v1, v2] += self._stress_tensor_contribution(
                        v1, v2, things, G1, G2, G_Gv, aa_xG, c_axi, Z_LvG)

        return stress_vv

    def _stress_tensor_contribution(self, v1, v2, things, G1, G2,
                                    G_Gv, a_xG, c_axi, Z_LvG):
        xp = self.xp
        f_IG = xp.empty((self.nI, G2 - G1), complex)
        emiGR_Ga = self.emiGR_Ga[G1:G2]
        Y_LG = self.Y_GL.T
        for a, l, I1, I2, f_G, dfdGoG_G in things:
            L1 = l**2
            L2 = (l + 1)**2
            f_IG[I1:I2] = (emiGR_Ga[:, a] * (-1.0j)**l *
                           (dfdGoG_G[G1:G2] * G_Gv[:, v1] * G_Gv[:, v2] *
                            Y_LG[L1:L2, G1:G2] +
                            f_G[G1:G2] * G_Gv[:, v1] * Z_LvG[L1:L2, v2]))

        c_xI = xp.zeros(a_xG.shape[:-1] + (self.nI,), self.pw.dtype)

        x = prod(c_xI.shape[:-1])
        if x == 0:
            return 0.0
        b_xI = c_xI.reshape((x, self.nI))
        a_xG = a_xG.reshape((x, a_xG.shape[-1]))

        alpha = 1.0
        if self.pw.dtype == float:
            alpha = 2.0
            if G1 == 0 and self.pw.comm.rank == 0:
                f_IG[:, 0] *= 0.5
            f_IG = f_IG.view(float)
            a_xG = a_xG.copy().view(float)

        if xp is np:
            mmm(alpha, a_xG, 'N', f_IG, 'C', 0.0, b_xI)
        else:
            xp.cublas.gemm('N', 'H', a_xG, f_IG, b_xI, alpha, 0.0)
        self.comm.sum(b_xI)

        stress = 0.0
        for a, I1, I2 in self.my_indices:
            stress -= self.eikR_a[a] * (c_axi[a] * c_xI[..., I1:I2]).sum()
        return stress.real
