from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

import numpy as np
from ase.units import Ha

import gpaw.fftw as fftw
from gpaw import debug
from gpaw.core.arrays import DistributedArrays
from gpaw.core.domain import Domain
from gpaw.core.matrix import Matrix
from gpaw.core.pwacf import PWAtomCenteredFunctions
from gpaw.gpu import cupy as cp
from gpaw.mpi import MPIComm, serial_comm
from gpaw.new import prod, zips
from gpaw.new.c import (add_to_density, add_to_density_gpu, pw_insert,
                        pw_insert_gpu)
from gpaw.pw.descriptor import pad
from gpaw.typing import (Array1D, Array2D, Array3D, ArrayLike1D, ArrayLike2D,
                         Literal, Vector)

if TYPE_CHECKING:
    from gpaw.core import UGArray, UGDesc


class PWDesc(Domain):
    itemsize = 16

    def __init__(self,
                 *,
                 ecut: float,  # hartree
                 cell: ArrayLike1D | ArrayLike2D,  # bohr
                 kpt: Vector | None = None,  # in units of reciprocal cell
                 comm: MPIComm = serial_comm,
                 dtype=None):
        """Description of plane-wave basis.

        parameters
        ----------
        ecut:
            Cutoff energy for kinetic energy of plane waves (units: hartree).
        cell:
            Unit cell given as three floats (orthorhombic grid), six floats
            (three lengths and the angles in degrees) or a 3x3 matrix
            (units: bohr).
        comm:
            Communicator for distribution of plane-waves.
        kpt:
            K-point for Block-boundary conditions specified in units of the
            reciprocal cell.
        dtype:
            Data-type (float or complex).
        """
        self.ecut = ecut
        Domain.__init__(self, cell, (True, True, True), kpt, comm, dtype)

        G_plus_k_Gv, ekin_G, self.indices_cG = find_reciprocal_vectors(
            ecut, self.cell_cv, self.kpt_c, self.dtype)

        # Find distribution:
        S = comm.size
        ng = len(ekin_G)
        self.maxmysize = (ng + S - 1) // S
        ng1 = comm.rank * self.maxmysize
        ng2 = min(ng1 + self.maxmysize, ng)
        self.ng1 = ng1
        self.ng2 = ng2

        # Distribute things:
        self.ekin_G = ekin_G[ng1:ng2].copy()
        self.ekin_G.flags.writeable = False
        # self.myindices_cG = self.indices_cG[:, ng1:ng2]
        self.G_plus_k_Gv = G_plus_k_Gv[ng1:ng2].copy()

        self.shape = (ng,)
        self.myshape = (len(self.ekin_G),)

        # Convert from np.float64 to float to avoid fake cupy problem ...
        # XXX Fix cpupy!!!
        self.dv = float(abs(np.linalg.det(self.cell_cv)))

        self._indices_cache: dict[tuple[int, ...], Array1D] = {}

    def __repr__(self) -> str:
        m = self.myshape[0]
        n = self.shape[0]
        return Domain.__repr__(self).replace(
            'Domain(',
            f'PWDesc(ecut={self.ecut} <coefs={m}/{n}>, ')

    def _short_string(self, global_shape):
        return (f'plane wave coefficients: {global_shape[-1]}\n'
                f'cutoff: {self.ecut * Ha} eV\n')

    def global_shape(self) -> tuple[int, ...]:
        """Tuple with one element: number of plane waves."""
        return self.shape

    def reciprocal_vectors(self) -> Array2D:
        """Returns reciprocal lattice vectors, G + k, in xyz coordinates."""
        return self.G_plus_k_Gv

    def kinetic_energies(self) -> Array1D:
        """Kinetic energy of plane waves.

        :::

             _ _ 2
            |G+k| / 2

        """
        return self.ekin_G

    def empty(self,
              dims: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm,
              xp=None) -> PWArray:
        """Create new PlaneWaveExpanions object.

        parameters
        ----------
        dims:
            Extra dimensions.
        comm:
            Distribute dimensions along this communicator.
        """
        return PWArray(self, dims, comm, xp=xp)

    def from_data(self, data):
        return PWArray(self, data.shape[:-1], data=data)

    def new(self,
            *,
            ecut: float | None = None,
            kpt=None,
            dtype=None,
            comm: MPIComm | Literal['inherit'] | None = 'inherit'
            ) -> PWDesc:
        """Create new plane-wave expansion description."""
        comm = self.comm if comm == 'inherit' else comm or serial_comm
        return PWDesc(ecut=ecut or self.ecut,
                      cell=self.cell_cv,
                      kpt=self.kpt_c if kpt is None else kpt,
                      dtype=dtype or self.dtype,
                      comm=comm or serial_comm)

    def indices(self, shape: tuple[int, ...]) -> Array1D:
        """Return indices into FFT-grid."""
        Q_G = self._indices_cache.get(shape)
        if Q_G is None:
            # We should do this here instead of everywhere calling this: !!!!
            # if self..dtype == float:
            #     shape = (shape[0], shape[1], shape[2] // 2 + 1)
            Q_G = np.ravel_multi_index(self.indices_cG, shape,  # type: ignore
                                       mode='wrap').astype(np.int32)
            self._indices_cache[shape] = Q_G
        return Q_G

    def cut(self, array_Q: Array3D) -> Array1D:
        """Cut out G-vectors with (G+k)^2/2<E_kin."""
        return array_Q.ravel()[self.indices(array_Q.shape)]

    def paste(self, coef_G: Array1D, array_Q: Array3D) -> None:
        """Paste G-vectors with (G+k)^2/2<E_kin into 3-D FFT grid and
        zero-pad."""
        Q_G = self.indices(array_Q.shape)
        if debug:
            assert (Q_G[1:] > Q_G[:-1]).all()
            assert (Q_G >= 0).all()
            assert (Q_G < array_Q.size).all()
            assert coef_G.shape == Q_G.shape
            assert coef_G.flags.c_contiguous
            assert Q_G.flags.c_contiguous
            assert array_Q.flags.c_contiguous

        pw_insert(coef_G, Q_G, 1.0, array_Q)

    def map_indices(self, other: PWDesc) -> tuple[Array1D, list[Array1D]]:
        """Map from one (distributed) set of plane waves to smaller global set.

        Say we have 9 G-vector on two cores::

           5 3 4             . 3 4           0 . .
           2 0 1 -> rank=0:  2 0 1  rank=1:  . . .
           8 6 7             . . .           3 1 2

        and we want a mapping to these 5 G-vectors::

             3
           2 0 1
             4

        On rank=0: the return values are::

           [0, 1, 2, 3], [[0, 1, 2, 3], [4]]

        and for rank=1::

           [1], [[0, 1, 2, 3], [4]]
        """
        size_c = tuple(np.ptp(self.indices_cG, axis=1) + 1)  # type: ignore
        Q_G = self.indices(size_c)
        G_Q = np.empty(prod(size_c), int)
        G_Q[Q_G] = np.arange(len(Q_G))
        G_g = G_Q[other.indices(size_c)]
        ng1 = 0
        g_r = []
        for rank in range(self.comm.size):
            ng2 = min(ng1 + self.maxmysize, self.shape[0])
            myg = (ng1 <= G_g) & (G_g < ng2)
            g_r.append(np.nonzero(myg)[0])
            if rank == self.comm.rank:
                my_G_g = G_g[myg] - ng1
            ng1 = ng2
        return my_G_g, g_r

    def atom_centered_functions(self,
                                functions,
                                positions,
                                *,
                                qspiral_v=None,
                                atomdist=None,
                                integral=None,
                                cut=False,
                                xp=None):
        """Create PlaneWaveAtomCenteredFunctions object."""
        if qspiral_v is None:
            return PWAtomCenteredFunctions(functions, positions, self,
                                           atomdist=atomdist,
                                           xp=xp)

        from gpaw.new.spinspiral import SpiralPWACF
        return SpiralPWACF(functions, positions, self,
                           atomdist=atomdist,
                           qspiral_v=qspiral_v)


class PWArray(DistributedArrays[PWDesc]):
    def __init__(self,
                 pw: PWDesc,
                 dims: int | tuple[int, ...] = (),
                 comm: MPIComm = serial_comm,
                 data: np.ndarray | None = None,
                 xp=None):
        """Object for storing function(s) as a plane-wave expansions.

        parameters
        ----------
        pw:
            Description of plane-waves.
        dims:
            Extra dimensions.
        comm:
            Distribute extra dimensions along this communicator.
        data:
            Data array for storage.
        """
        DistributedArrays. __init__(self, dims, pw.myshape,
                                    comm, pw.comm,
                                    data, pw.dv, complex, xp)
        self.desc = pw
        self._matrix: Matrix | None

    def __repr__(self):
        txt = f'PWArray(pw={self.desc}, dims={self.dims}'
        if self.comm.size > 1:
            txt += f', comm={self.comm.rank}/{self.comm.size}'
        if self.xp is not np:
            txt += ', xp=cp'
        return txt + ')'

    def __getitem__(self, index: int | slice) -> PWArray:
        data = self.data[index]
        return PWArray(self.desc,
                       data.shape[:-1],
                       data=data)

    def __iter__(self):
        for data in self.data:
            yield PWArray(self.desc,
                          data.shape[:-1],
                          data=data)

    def new(self,
            data=None) -> PWArray:
        """Create new PWArray object of same kind.

        Parameters
        ----------
        data:
            Array to use for storage.
        """
        if data is None:
            data = self.xp.empty_like(self.data)
        else:
            # Number of plane-waves depends on the k-point.  We therfore
            # allow for data to be bigger than needed:
            data = data.ravel()[:self.data.size].reshape(self.data.shape)
        return PWArray(self.desc,
                       self.dims,
                       self.comm,
                       data)

    def copy(self):
        """Create a copy (surprise!)."""
        a = self.new()
        a.data[:] = self.data
        return a

    def sanity_check(self) -> None:
        """Sanity check for real-valed PW expansions.

        Make sure the G=(0,0,0) coefficient doesn't have an imaginary part.
        """
        if self.desc.dtype == float and self.desc.comm.rank == 0:
            if (self.data[..., 0].imag != 0.0).any():
                raise ValueError

    def _arrays(self):
        shape = self.data.shape
        return self.data.reshape((prod(shape[:-1]), shape[-1]))

    @property
    def matrix(self) -> Matrix:
        """Matrix view of data."""
        if self._matrix is not None:
            return self._matrix

        shape = (self.dims[0], prod(self.dims[1:]) * self.myshape[0])
        myshape = (self.mydims[0], prod(self.mydims[1:]) * self.myshape[0])
        dist = (self.comm, -1, 1)
        data = self.data.reshape(myshape)

        if self.desc.dtype == float:
            data = data.view(float)
            shape = (shape[0], shape[1] * 2)

        self._matrix = Matrix(*shape, data=data, dist=dist)
        return self._matrix

    def ifft(self, *, plan=None, grid=None, out=None, periodic=False):
        """Do inverse FFT(s) to uniform grid(s).

        Parameters
        ----------
        plan:
            Plan for inverse FFT.
        grid:
            Target grid.
        out:
            Target UGArray object.
        """
        comm = self.desc.comm
        xp = self.xp
        if out is None:
            out = grid.empty(self.dims, xp=xp)
        assert self.desc.dtype == out.desc.dtype, (self.desc, out.desc)
        assert not out.desc.zerobc_c.any()
        assert comm.size == out.desc.comm.size, (comm, out.desc.comm)

        plan = plan or out.desc.fft_plans(xp=xp)
        this = self.gather()
        if this is not None:
            for coef_G, out1 in zips(this._arrays(), out.flat()):
                plan.ifft_sphere(coef_G, self.desc, out1)
        else:
            for out1 in out.flat():
                plan.ifft_sphere(None, self.desc, out1)

        if not periodic:
            out.multiply_by_eikr()

        return out

    def interpolate(self,
                    plan1: fftw.FFTPlans | None = None,
                    plan2: fftw.FFTPlans | None = None,
                    grid: UGDesc | None = None,
                    out: UGArray | None = None) -> UGArray:
        assert plan1 is None
        return self.ifft(plan=plan2, grid=grid, out=out)

    def gather(self, out=None, broadcast=False):
        """Gather coefficients on master."""
        comm = self.desc.comm

        if comm.size == 1:
            if out is None:
                return self
            out.data[:] = self.data
            return out

        if out is None:
            if comm.rank == 0 or broadcast:
                pw = self.desc.new(comm=serial_comm)
                out = pw.empty(self.dims, comm=self.comm, xp=self.xp)
            else:
                out = Empty(self.mydims)

        if comm.rank == 0:
            data = self.xp.empty(self.desc.maxmysize * comm.size, complex)
        else:
            data = None

        for input, output in zips(self._arrays(), out._arrays()):
            mydata = pad(input, self.desc.maxmysize)
            comm.gather(mydata, 0, data)
            if comm.rank == 0:
                output[:] = data[:len(output)]

        if broadcast:
            comm.broadcast(out.data, 0)

        return out if not isinstance(out, Empty) else None

    def gather_all(self, out: PWArray) -> None:
        """Gather coefficients from self[r] on rank r.

        On rank r, an array of all G-vector coefficients will be returned.
        These will be gathered from self[r] on all the cores.
        """
        assert len(self.dims) == 1
        pw = self.desc
        comm = pw.comm
        if comm.size == 1:
            out.data[:] = self.data[0]
            return

        N = self.dims[0]
        assert N <= comm.size

        ng = pw.shape[0]
        myng = pw.myshape[0]
        maxmyng = pw.maxmysize

        ssize_r, soffset_r, rsize_r, roffset_r = a2a_stuff(
            comm, N, ng, myng, maxmyng)

        comm.alltoallv(self.data, ssize_r, soffset_r,
                       out.data, rsize_r, roffset_r)

    def scatter_from(self, data: Array1D | PWArray | None = None) -> None:
        """Scatter data from rank-0 to all ranks."""
        if isinstance(data, PWArray):
            data = data.data
        comm = self.desc.comm
        if comm.size == 1:
            assert data is not None
            self.data[:] = self.xp.asarray(data)
            return

        if comm.rank == 0:
            assert data is not None
            shape = data.shape
            for fro, to in zips(data.reshape((prod(shape[:-1]), shape[-1])),
                                self._arrays()):
                fro = pad(fro, comm.size * self.desc.maxmysize)
                comm.scatter(fro, to, 0)
        else:
            buf = self.xp.empty(self.desc.maxmysize, complex)
            for to in self._arrays():
                comm.scatter(None, buf, 0)
                to[:] = buf[:len(to)]

    def scatter_from_all(self, a_G: PWArray) -> None:
        """Scatter all coefficients from rank r to self on other cores."""
        assert len(self.dims) == 1
        pw = self.desc
        comm = pw.comm
        if comm.size == 1:
            self.data[:] = a_G.data
            return

        N = self.dims[0]
        assert N <= comm.size

        ng = pw.shape[0]
        myng = pw.myshape[0]
        maxmyng = pw.maxmysize

        rsize_r, roffset_r, ssize_r, soffset_r = a2a_stuff(
            comm, N, ng, myng, maxmyng)

        comm.alltoallv(a_G.data, ssize_r, soffset_r,
                       self.data, rsize_r, roffset_r)

    def integrate(self, other: PWArray | None = None) -> np.ndarray:
        """Integral of self or self time cc(other)."""
        dv = self.dv
        if other is not None:
            assert self.comm.size == 1
            assert self.desc.dtype == other.desc.dtype
            a = self._arrays()
            b = other._arrays()
            if self.desc.dtype == float:
                a = a.view(float)
                b = b.view(float)
                dv *= 2
            result = a @ b.T.conj()
            if self.desc.dtype == float and self.desc.comm.rank == 0:
                result -= 0.5 * a[:, :1] @ b[:, :1].T
            self.desc.comm.sum(result)
            result = result.reshape(self.dims + other.dims)
        else:
            if self.desc.comm.rank == 0:
                result = self.data[..., 0]
            else:
                result = self.xp.empty(self.mydims, complex)
            self.desc.comm.broadcast(result, 0)
        if self.desc.dtype == float:
            result = result.real
        if result.ndim == 0:
            result = result.item()  # convert to scalar
        return result * dv

    def _matrix_elements_correction(self,
                                    M1: Matrix,
                                    M2: Matrix,
                                    out: Matrix,
                                    symmetric: bool) -> None:
        if self.desc.dtype == float:
            out.data *= 2.0
            if self.desc.comm.rank == 0:
                correction = M1.data[:, :1] @ M2.data[:, :1].T
                if symmetric:
                    correction *= 0.5 * self.dv
                    out.data -= correction
                    out.data -= correction.T
                else:
                    correction *= self.dv
                    out.data -= correction

    def norm2(self, kind: str = 'normal') -> np.ndarray:
        r"""Calculate integral over cell.

        For kind='normal' we calculate:::

          /   _  2 _   --    2
          ||a(r)| dr = > |c | V,
          /            --  G
                        G

        where V is the volume of the unit cell.

        And for kind='kinetic':::

           1  --    2  2
          --- > |c |  G V,
           2  --  G
               G

        """
        a_xG = self._arrays().view(float)
        if kind == 'normal':
            result_x = self.xp.einsum('xG, xG -> x', a_xG, a_xG)
        elif kind == 'kinetic':
            x, G2 = a_xG.shape
            a_xGz = a_xG.reshape((x, G2 // 2, 2))
            result_x = self.xp.einsum('xGz, xGz, G -> x',
                                      a_xGz,
                                      a_xGz,
                                      self.xp.asarray(self.desc.ekin_G))
        else:
            1 / 0
        if self.desc.dtype == float:
            result_x *= 2
            if self.desc.comm.rank == 0 and kind == 'normal':
                result_x -= a_xG[:, 0]**2
        self.desc.comm.sum(result_x)
        return result_x.reshape(self.mydims) * self.dv

    def abs_square(self,
                   weights: Array1D,
                   out: UGArray,
                   _slow: bool = False) -> None:
        """Add weighted absolute square of self to output array.

        With `a_n(G)` being self and `w_n` the weights:::

              _         _    --     -1    _   2
          out(r) <- out(r) + >  |FFT  [a (G)]| w
                             --         n       n
                             n

        """
        pw = self.desc
        domain_comm = pw.comm
        xp = self.xp
        a_nG = self

        if domain_comm.size == 1:
            if not _slow and xp is cp and pw.dtype == complex:
                return abs_square_gpu(a_nG, weights, out)

            a_R = out.desc.new(dtype=pw.dtype).empty(xp=xp)
            for weight, a_G in zips(weights, a_nG):
                if weight == 0.0:
                    continue
                a_G.ifft(out=a_R)
                if xp is np:
                    add_to_density(weight, a_R.data, out.data)
                else:
                    out.data += float(weight) * xp.abs(a_R.data)**2
            return

        # Undistributed work arrays:
        a1_R = out.desc.new(comm=None, dtype=pw.dtype).empty(xp=xp)
        a1_G = pw.new(comm=None).empty(xp=xp)
        b1_R = out.desc.new(comm=None).zeros(xp=xp)

        (N,) = self.mydims
        for n1 in range(0, N, domain_comm.size):
            n2 = min(n1 + domain_comm.size, N)
            a_nG[n1:n2].gather_all(a1_G)
            n = n1 + domain_comm.rank
            if n >= N:
                continue
            weight = weights[n]
            if weight == 0.0:
                continue
            a1_G.ifft(out=a1_R)
            if xp is np:
                add_to_density(weight, a1_R.data, b1_R.data)
            else:
                b1_R.data += float(weight) * xp.abs(a1_R.data)**2

        domain_comm.sum(b1_R.data)
        b_R = out.new()
        b_R.scatter_from(b1_R)
        out.data += b_R.data

    def to_pbc_grid(self):
        return self

    def randomize(self, seed: int | None = None) -> None:
        """Insert random numbers between -0.5 and 0.5 into data."""
        if seed is None:
            seed = self.comm.rank + self.desc.comm.rank * self.comm.size
        rng = self.xp.random.default_rng(seed)
        a = self.data.view(float)
        rng.random(a.shape, out=a)
        a -= 0.5
        if self.desc.dtype == float and self.desc.comm.rank == 0:
            a[..., 1] = 0.0

    def moment(self):
        pw = self.desc
        # Masks:
        m0_G, m1_G, m2_G = (i_G == 0 for i_G in pw.indices_cG)
        a_G = self.gather()
        if a_G is not None:
            b_G = a_G.data.imag
            b_cs = [b_G[m1_G & m2_G],
                    b_G[m0_G & m2_G],
                    b_G[m0_G & m1_G]]
            d_c = [b_s[1:] @ (1.0 / np.arange(1, len(b_s)))
                   for b_s in b_cs]
            m_v = np.dot(d_c, pw.cell_cv) / pi * pw.dv
        else:
            m_v = np.empty(3)
        pw.comm.broadcast(m_v, 0)
        return m_v

    def morph(self, pw: PWDesc) -> PWArray:
        pw0 = self.desc
        out_xG = pw.zeros(self.dims,
                          comm=self.comm,
                          xp=self.xp)
        assert isinstance(out_xG, PWArray)  # MYPY!!!!

        d = {}
        for G, i_c in enumerate(pw.indices_cG.T):
            d[tuple(i_c)] = G
        G_G0 = []
        G0_G = []
        for G0, i_c in enumerate(pw0.indices_cG.T):
            G = d.get(tuple(i_c), -1)
            if G != -1:
                G_G0.append(G)
                G0_G.append(G0)

        out_xG.data[..., G_G0] = self.data[..., G0_G]
        return out_xG

    def add_ked(self,
                occ_n: Array1D,
                taut_R: UGArray) -> None:
        psit_nG = self
        pw = psit_nG.desc
        domain_comm = pw.comm

        # Undistributed work arrays:
        dpsit1_R = taut_R.desc.new(comm=None, dtype=pw.dtype).empty()
        pw1 = pw.new(comm=None)
        psit1_G = pw1.empty()
        iGpsit1_G = pw1.empty()
        taut1_R = taut_R.desc.new(comm=None).zeros()
        Gplusk1_Gv = pw1.reciprocal_vectors()

        (N,) = psit_nG.mydims
        for n1 in range(0, N, domain_comm.size):
            n2 = min(n1 + domain_comm.size, N)
            psit_nG[n1:n2].gather_all(psit1_G)
            n = n1 + domain_comm.rank
            if n >= N:
                continue
            f = occ_n[n]
            if f == 0.0:
                continue
            for v in range(3):
                iGpsit1_G.data[:] = psit1_G.data
                iGpsit1_G.data *= 1j * Gplusk1_Gv[:, v]
                iGpsit1_G.ifft(out=dpsit1_R)
                add_to_density(0.5 * f, dpsit1_R.data, taut1_R.data)
        domain_comm.sum(taut1_R.data)
        tmp_R = taut_R.new()
        tmp_R.scatter_from(taut1_R)
        taut_R.data += tmp_R.data


def a2a_stuff(comm, N, ng, myng, maxmyng):
    """Create arrays for MPI alltoallv call."""
    ssize_r = np.zeros(comm.size, int)
    ssize_r[:N] = myng
    soffset_r = np.arange(comm.size) * myng
    soffset_r[N:] = 0
    roffset_r = (np.arange(comm.size) * maxmyng).clip(max=ng)
    rsize_r = np.zeros(comm.size, int)
    if comm.rank < N:
        rsize_r[:-1] = roffset_r[1:] - roffset_r[:-1]
        rsize_r[-1] = ng - roffset_r[-1]
    return ssize_r, soffset_r, rsize_r, roffset_r


class Empty:
    def __init__(self, dims):
        self.dims = dims

    def _arrays(self):
        for _ in range(prod(self.dims)):
            yield


def find_reciprocal_vectors(ecut: float,
                            cell: Array2D,
                            kpt=np.zeros(3),
                            dtype=complex) -> tuple[Array2D,
                                                    Array1D,
                                                    Array2D]:
    """Find reciprocal lattice vectors inside sphere.

    >>> cell = np.eye(3)
    >>> ecut = 0.5 * (2 * pi)**2
    >>> G, e, i = find_reciprocal_vectors(ecut, cell)
    >>> G
    array([[ 0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  6.28318531],
           [ 0.        ,  0.        , -6.28318531],
           [ 0.        ,  6.28318531,  0.        ],
           [ 0.        , -6.28318531,  0.        ],
           [ 6.28318531,  0.        ,  0.        ],
           [-6.28318531,  0.        ,  0.        ]])
    >>> e
    array([ 0.       , 19.7392088, 19.7392088, 19.7392088, 19.7392088,
           19.7392088, 19.7392088])
    >>> i
    array([[ 0,  0,  0,  0,  0,  1, -1],
           [ 0,  0,  0,  1, -1,  0,  0],
           [ 0,  1, -1,  0,  0,  0,  0]])
    """
    Gcut = (2 * ecut)**0.5
    n = Gcut * (cell**2).sum(axis=1)**0.5 / (2 * pi) + abs(kpt)
    size = 2 * n.astype(int) + 4

    if dtype == float:
        size[2] = size[2] // 2 + 1
        i_Qc = np.indices(size).transpose((1, 2, 3, 0))
        i_Qc[..., :2] += size[:2] // 2
        i_Qc[..., :2] %= size[:2]
        i_Qc[..., :2] -= size[:2] // 2
    else:
        i_Qc = np.indices(size).transpose((1, 2, 3, 0))  # type: ignore
        half = [s // 2 for s in size]
        i_Qc += half
        i_Qc %= size
        i_Qc -= half

    # Calculate reciprocal lattice vectors:
    B_cv = 2.0 * pi * np.linalg.inv(cell).T
    # i_Qc.shape = (-1, 3)
    G_plus_k_Qv = (i_Qc + kpt) @ B_cv

    ekin = 0.5 * (G_plus_k_Qv**2).sum(axis=3)
    mask = ekin <= ecut

    assert not mask[size[0] // 2].any()
    assert not mask[:, size[1] // 2].any()
    if dtype == complex:
        assert not mask[:, :, size[2] // 2].any()
    else:
        assert not mask[:, :, -1].any()

    if dtype == float:
        mask &= ((i_Qc[..., 2] > 0) |
                 (i_Qc[..., 1] > 0) |
                 ((i_Qc[..., 0] >= 0) & (i_Qc[..., 1] == 0)))

    indices = i_Qc[mask]
    ekin = ekin[mask]
    G_plus_k = G_plus_k_Qv[mask]

    return G_plus_k, ekin, indices.T


def abs_square_gpu(psit_nG, weight_n, nt_R):
    from gpaw.gpu import cupyx
    pw = psit_nG.desc
    plan = nt_R.desc.fft_plans(xp=cp, dtype=complex)
    Q_G = cp.asarray(plan.indices(pw))
    weight_n = cp.asarray(weight_n)
    N = len(weight_n)
    shape = tuple(nt_R.desc.size_c)
    B = 32
    psit_bR = None
    for b1 in range(0, N, B):
        b2 = min(b1 + B, N)
        nb = b2 - b1
        if psit_bR is None:
            psit_bR = cp.empty((nb,) + shape, complex)
        elif nb < B:
            psit_bR = psit_bR[:nb]
        psit_bR[:] = 0.0
        pw_insert_gpu(psit_nG.data[b1:b2],
                      Q_G,
                      1.0,
                      psit_bR.reshape((nb, -1)))
        psit_bR[:] = cupyx.scipy.fft.ifftn(
            psit_bR,
            shape,
            norm='forward',
            overwrite_x=True)
        add_to_density_gpu(weight_n[b1:b2], psit_bR, nt_R.data)
