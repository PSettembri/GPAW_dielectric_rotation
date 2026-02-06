import numpy as np

import ase.units as units
from ase.parallel import parprint
from ase.phonons import Displacement
from ase.utils.filecache import MultiFileJSONCache

from gpaw.calculator import GPAW
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.lcao.tightbinding import TightBinding
from gpaw.utilities import unpack_hermitian
from gpaw.utilities.tools import tri2full


# TODO replace parprint with something nicer
class ElectronPhononCoupling(Displacement):
    """Class for calculating the electron-phonon coupling in an LCAO basis.

    The derivative of the effective potential wrt atomic displacements is
    obtained from a finite difference approximation to the derivative by doing
    a self-consistent calculation for atomic displacements in the +/-
    directions. These calculations are carried out in the ``run`` member
    function.

    The subsequent calculation of the coupling matrix in the basis of atomic
    orbitals (or Bloch-sums hereof for periodic systems) is handled by the
    ``calculate_matrix`` member function.

    """

    def __init__(self, atoms, calc=None, supercell=(1, 1, 1), name='elph',
                 delta=0.01, calculate_forces=False):
        """Initialize with base class args and kwargs.

        Parameters
        ----------
        atoms: Atoms
            The atoms to work on.
        calc: GPAW
            Calculator for the supercell finite displacement calculation.
        supercell: tuple, list
            Size of supercell given by the number of repetitions (l, m, n) of
            the small unit cell in each direction.
        name: str
            Name to use for files (default: 'elph').
        delta: float
            Magnitude of displacements.
        calculate_forces: bool
            If true, also calculate and store the dynamical matrix.
        """

        parprint("DEPRECATION WARNING: This module is deprecated.")

        # Init base class and make the center cell in the supercell the
        # reference cell
        Displacement.__init__(self, atoms, calc=calc, supercell=supercell,
                              name=name, delta=delta, center_refcell=True)

        self.calculate_forces = calculate_forces
        # LCAO calculator
        self.calc_lcao = None
        # Supercell matrix
        self.g_xsNNMM = None
        self.basis_info = None
        self.supercell_cache = None

    def calculate(self, atoms_N, disp):
        return self(atoms_N)

    def __call__(self, atoms_N):
        """Extract effective potential and projector coefficients."""

        # Do calculation
        atoms_N.get_potential_energy()

        # Calculate forces if desired
        if self.calculate_forces:
            forces = atoms_N.get_forces()
        else:
            forces = None

        # Get calculator
        calc = atoms_N.calc
        if not isinstance(calc, GPAW):
            calc = calc.dft  # unwrap DFTD3 wrapper

        # Effective potential (in Hartree) and projector coefficients
        Vt_sG = calc.hamiltonian.vt_sG
        Vt_sG = calc.wfs.gd.collect(Vt_sG, broadcast=True)
        dH_asp = calc.hamiltonian.dH_asp

        setups = calc.wfs.setups
        nspins = calc.wfs.nspins
        gd_comm = calc.wfs.gd.comm

        dH_all_asp = {}
        for a, setup in enumerate(setups):
            ni = setup.ni
            nii = ni * (ni + 1) // 2
            dH_tmp_sp = np.zeros((nspins, nii))
            if a in dH_asp:
                dH_tmp_sp[:] = dH_asp[a]
            gd_comm.sum(dH_tmp_sp)
            dH_all_asp[a] = dH_tmp_sp

        output = {'Vt_sG': Vt_sG, 'dH_all_asp': dH_all_asp}
        if forces is not None:
            output['forces'] = forces
        return output

    def set_lcao_calculator(self, calc):
        """Set LCAO calculator for the calculation of the supercell matrix."""
        assert calc.wfs.mode == 'lcao', 'LCAO mode required.'
        assert not calc.symmetry.point_group, \
            'Point group symmetry not supported'
        self.calc_lcao = calc

    def set_basis_info(self, *args):
        """Store lcao basis info for atoms in reference cell in attribute.

        Parameters
        ----------
        args: tuple
            If the LCAO calculator is not available (e.g. if the supercell is
            loaded from file), the ``load_supercell_matrix`` member function
            provides the required info as arguments.

        """
        assert len(args) in (0, 2)
        if len(args) == 0:
            calc = self.calc_lcao
            setups = calc.wfs.setups
            bfs = calc.wfs.basis_functions
            nao_a = [setups[a].nao for a in range(len(self.atoms))]
            M_a = [bfs.M_a[a] for a in range(len(self.atoms))]
        else:
            M_a = args[0]
            nao_a = args[1]

        self.basis_info = {'M_a': M_a, 'nao_a': nao_a}

    def _calculate_supercell_entry(self, a, v, V1t_sG, dH1_asp, wfs,
                                   include_pseudo):
        kpt_u = wfs.kpt_u
        setups = wfs.setups
        nao = setups.nao
        bfs = wfs.basis_functions
        dtype = wfs.dtype
        nspins = wfs.nspins

        # Equilibrium atomic Hamiltonian matrix (projector coefficients)
        dH_asp = self.cache['eq']['dH_all_asp']

        # For the contribution from the derivative of the projectors
        dP_aqvMi = wfs.manytci.P_aqMi(self.indices, derivative=True)
        dP_qvMi = dP_aqvMi[a]

        # Array for different k-point components
        g_sqMM = np.zeros((nspins, len(kpt_u) // nspins, nao, nao), dtype)

        # 1) Gradient of effective potential
        parprint("Starting gradient of effective potential")
        for kpt in kpt_u:
            # Matrix elements
            geff_MM = np.zeros((nao, nao), dtype)
            bfs.calculate_potential_matrix(V1t_sG[kpt.s], geff_MM, q=kpt.q)
            tri2full(geff_MM, 'L')
            # Insert in array
            g_sqMM[kpt.s, kpt.q] += geff_MM
        parprint("Finished gradient of effective potential")

        if include_pseudo:
            parprint("Starting gradient of pseudo part")
            # 2) Gradient of non-local part (projectors)
            # parprint("Starting gradient of dH^a")
            P_aqMi = wfs.P_aqMi
            # 2a) dH^a part has contributions from all other atoms
            for kpt in kpt_u:
                # Matrix elements
                gp_MM = np.zeros((nao, nao), dtype)
                for a_, dH1_sp in dH1_asp.items():
                    dH1_ii = unpack_hermitian(dH1_sp[kpt.s])
                    P_Mi = P_aqMi[a_][kpt.q]
                    gp_MM += np.dot(P_Mi, np.dot(dH1_ii, P_Mi.T.conjugate()))
                g_sqMM[kpt.s, kpt.q] += gp_MM
            # parprint("Finished gradient of dH^a")

            # parprint("Starting gradient of projectors")
            # 2b) dP^a part has only contributions from the same atoms
            dH_ii = unpack_hermitian(dH_asp[a][kpt.s])
            for kpt in kpt_u:
                # XXX Sort out the sign here; conclusion -> sign = +1 !
                P1HP_MM = +1 * np.dot(dP_qvMi[kpt.q][v], np.dot(dH_ii,
                                      P_aqMi[a][kpt.q].T.conjugate()))
                # Matrix elements
                gp_MM = P1HP_MM + P1HP_MM.T.conjugate()
                g_sqMM[kpt.s, kpt.q] += gp_MM
            # parprint("Finished gradient of projectors")
            parprint("Finished gradient of pseudo part")
        return g_sqMM

    def calculate_supercell_matrix(self, name='supercell', filter=None,
                                   include_pseudo=True):
        """Calculate matrix elements of the el-ph coupling in the LCAO basis.

        This function calculates the matrix elements between LCAOs and local
        atomic gradients of the effective potential. The matrix elements are
        calculated for the supercell used to obtain finite-difference
        approximations to the derivatives of the effective potential wrt to
        atomic displacements.

        Parameters
        ----------
        name: str
            User specified name of the generated JSON cache.
            Default is 'supercell'.
        filter: str
            Fourier filter atomic gradients of the effective potential. The
            specified components (``normal`` or ``umklapp``) are removed
            (default: None).
        include_pseudo: bool
            Include the contribution from the psedupotential in the atomic
            gradients. If ``False``, only the gradient of the effective
            potential is included (default: True).
        """

        assert self.calc_lcao is not None, "Set LCAO calculator"

        # JSON cache
        self.supercell_cache = MultiFileJSONCache(name)

        # Supercell atoms
        atoms_N = self.atoms * self.supercell

        # Initialize calculator if required and extract useful quantities
        calc = self.calc_lcao
        if (not hasattr(calc.wfs, 'S_qMM') or
            not hasattr(calc.wfs.basis_functions, 'M_a')):
            calc.initialize(atoms_N)
            calc.initialize_positions(atoms_N)
        self.set_basis_info()

        # Extract useful objects from the calculator
        wfs = calc.wfs
        gd = calc.wfs.gd
        kd = calc.wfs.kd
        setups = wfs.setups
        nao = setups.nao
        nspins = wfs.nspins
        # FIXME: Domain parallelisation broken
        assert gd.comm.size == 1

        # If gamma calculation, overlap with neighboring cell cannot be removed
        if kd.gamma:
            print("WARNING: Gamma-point calculation.")
        else:
            # Bloch to real-space converter
            tb = TightBinding(atoms_N, calc)

        parprint("Calculating supercell matrix")

        parprint("Calculating real-space gradients")
        # Calculate finite-difference gradients (in Hartree / Bohr)
        V1t_xsG, dH1_xasp = self.calculate_gradient()
        parprint("Finished real-space gradients")

        # Fourier filter the atomic gradients of the effective potential
        if filter is not None:
            parprint("Fourier filtering gradients")
            for s in range(nspins):
                self.fourier_filter(V1t_xsG[:, s], components=filter)
            parprint("Finished Fourier filtering")

        # Check that the grid is the same as in the calculator
        assert np.all(V1t_xsG.shape[-3:] == (gd.N_c + gd.pbc_c - 1)), \
            "Mismatch in grids."

        with self.supercell_cache.lock('basis') as handle:
            if handle is not None:
                handle.save(self.basis_info)

        # Calculate < i k | grad H | j k >, i.e. matrix elements in Bloch basis
        parprint("Calculating gradient of PAW Hamiltonian")

        # Do each cartesian component separately
        for i, a in enumerate(self.indices):
            for v in range(3):
                # Corresponding array index
                x = 3 * i + v

                # If exist already, don't recompute
                with self.supercell_cache.lock(str(x)) as handle:
                    if handle is None:
                        continue

                    # parprint("Atom ", i+1, "/", len(self.indices), " ,
                    # direction ", v)
                    parprint("%s-gradient of atom %u" %
                             (['x', 'y', 'z'][v], a))

                    g_sqMM = self._calculate_supercell_entry(a, v, V1t_xsG[x],
                                                             dH1_xasp[x], wfs,
                                                             include_pseudo)

                    # Extract R_c=(0, 0, 0) block by Fourier transforming
                    if kd.gamma or kd.N_c is None:
                        g_sMM = g_sqMM[:, 0]
                    else:
                        # Convert to array
                        g_sMM = []
                        for s in range(nspins):
                            g_MM = tb.bloch_to_real_space(g_sqMM[s],
                                                          R_c=(0, 0, 0))
                            g_sMM.append(g_MM[0])  # [0] because of above
                        g_sMM = np.array(g_sMM)

                    # Reshape to global unit cell indices
                    N = np.prod(self.supercell)
                    # Number of basis function in the primitive cell
                    assert (nao % N) == 0, "Alarm ...!"
                    nao_cell = nao // N
                    g_sNMNM = g_sMM.reshape((nspins, N, nao_cell, N, nao_cell))
                    g_sNNMM = g_sNMNM.swapaxes(2, 3).copy()
                    parprint("Finished supercell matrix")

                    handle.save(g_sNNMM)
                if x == 0:
                    with self.supercell_cache.lock('info') as handle:
                        if handle is not None:
                            handle.save([g_sNNMM.shape, g_sNNMM.dtype.name])
        parprint("Finished gradient of PAW Hamiltonian")

    def load_supercell_matrix(self, name='supercell'):
        """Load supercell matrix from cache.

        Parameters
        ----------
        name: str
            User specified name of the cache.
        """
        self.supercell_cache = MultiFileJSONCache(name)
        self.basis_info = self.supercell_cache['basis']
        [shape, dtype] = self.supercell_cache['info']

        nx = len(self.indices) * 3
        self.g_xsNNMM = np.empty([nx, ] + list(shape), dtype=dtype)
        for x in range(nx):
            self.g_xsNNMM[x] = self.supercell_cache[str(x)]

    def apply_cutoff(self, cutmax=None, cutmin=None):
        """Zero matrix element inside/beyond the specified cutoffs.

        This method is not tested.
        This method does not respect minimum image convention.

        Parameters
        ----------
        cutmax: float
            Zero matrix elements for basis functions with a distance to the
            atomic gradient that is larger than the cutoff.
        cutmin: float
            Zero matrix elements where both basis functions have distances to
            the atomic gradient that is smaller than the cutoff.
        """

        if cutmax is not None:
            cutmax = float(cutmax)
        if cutmin is not None:
            cutmin = float(cutmin)

        # Reference to supercell matrix attribute
        g_xsNNMM = self.g_xsNNMM

        # Number of atoms and primitive cells
        N_atoms = len(self.indices)
        N = np.prod(self.supercell)
        nao = g_xsNNMM.shape[-1]
        nspins = g_xsNNMM.shape[1]

        # Reshape array
        g_avsNNMM = g_xsNNMM.reshape(N_atoms, 3, nspins, N, N, nao, nao)

        # Make slices for orbitals on atoms
        M_a = self.basis_info['M_a']
        nao_a = self.basis_info['nao_a']
        slice_a = []
        for a in range(len(self.atoms)):
            start = M_a[a]
            stop = start + nao_a[a]
            s = slice(start, stop)
            slice_a.append(s)

        # Lattice vectors
        R_cN = self.compute_lattice_vectors()

        # Unit cell vectors
        cell_vc = self.atoms.cell.transpose()
        # Atomic positions in reference cell
        pos_av = self.atoms.get_positions()

        # Create a mask array to zero the relevant matrix elements
        if cutmin is not None:
            mask_avsNNMM = np.zeros(g_avsNNMM.shape, dtype=bool)

        # Zero elements where one of the basis orbitals has a distance to atoms
        # (atomic gradients) in the reference cell larger than the cutoff
        for n in range(N):
            # Lattice vector to cell
            R_v = np.dot(cell_vc, R_cN[:, n])
            # Atomic positions in cell
            posn_av = pos_av + R_v
            for i, a in enumerate(self.indices):
                # Atomic distances wrt to the position of the gradient
                dist_a = np.sqrt(np.sum((pos_av[a] - posn_av)**2, axis=-1))

                if cutmax is not None:
                    # Atoms indices where the distance is larger than the max
                    # cufoff
                    j_a = np.where(dist_a > cutmax)[0]
                    # Zero elements
                    for j in j_a:
                        g_avsNNMM[a, :, :, n, :, slice_a[j], :] = 0.0
                        g_avsNNMM[a, :, :, :, n, :, slice_a[j]] = 0.0

                if cutmin is not None:
                    # Atoms indices where the distance is larger than the min
                    # cufoff
                    j_a = np.where(dist_a > cutmin)[0]
                    # Update mask to keep elements where one LCAO is outside
                    # the min cutoff
                    for j in j_a:
                        mask_avsNNMM[a, :, :, n, :, slice_a[j], :] = True
                        mask_avsNNMM[a, :, :, :, n, :, slice_a[j]] = True

        # Zero elements where both LCAOs are located within the min cutoff
        if cutmin is not None:
            g_avsNNMM[~mask_avsNNMM] = 0.0

    def lcao_matrix(self, u_l, omega_l):
        """Calculate the el-ph coupling in the electronic LCAO basis.

        For now, only works for Gamma-point phonons.

        This method is not tested.

        Parameters
        ----------
        u_l: np.ndarray
            Mass-scaled polarization vectors (in units of 1 / sqrt(amu)) of the
            phonons.
        omega_l: np.ndarray
            Vibrational frequencies in eV.
        """

        # Supercell matrix (Hartree / Bohr)
        assert self.g_xsNNMM is not None, "Load supercell matrix."
        assert self.g_xsNNMM.shape[2:4] == (1, 1)
        g_xsMM = self.g_xsNNMM[:, :, 0, 0, :, :]
        # Number of atomic orbitals
        # nao = g_xMM.shape[-1]
        # Number of phonon modes
        nmodes = u_l.shape[0]

        #
        u_lx = u_l.reshape(nmodes, 3 * len(self.atoms))
        # np.dot uses second to last index of second array
        g_lsMM = np.dot(u_lx, g_xsMM.transpose(2, 0, 1, 3))

        # Multiply prefactor sqrt(hbar / 2 * M * omega) in units of Bohr
        amu = units._amu  # atomic mass unit
        me = units._me   # electron mass
        g_lsMM /= np.sqrt(2 * amu / me / units.Hartree *
                          omega_l[:, :, np.newaxis, np.newaxis])
        # Convert to eV
        g_lsMM *= units.Hartree

        return g_lsMM

    def bloch_matrix(self, kpts, qpts, c_kn, u_ql,
                     omega_ql=None, kpts_from=None, spin=0,
                     name='supercell'):
        r"""Calculate el-ph coupling in the Bloch basis for the electrons.

        This function calculates the electron-phonon coupling between the
        specified Bloch states, i.e.::

                      ______
            mnl      / hbar               ^
           g    =   /-------  < m k + q | e  . grad V  | n k >
            kq    \/ 2 M w                 ql        q
                          ql

        In case the ``omega_ql`` keyword argument is not given, the bare matrix
        element (in units of eV / Ang) without the sqrt prefactor is returned.

        Phonon frequencies and mode vectors must be given in
        ase units.

        Parameters
        ----------
        kpts: np.ndarray or tuple
            k-vectors of the Bloch states. When a tuple of integers is given, a
            Monkhorst-Pack grid with the specified number of k-points along the
            directions of the reciprocal lattice vectors is generated.
        qpts: np.ndarray or tuple
            q-vectors of the phonons.
        c_kn: np.ndarray
            Expansion coefficients for the Bloch states. The ordering must be
            the same as in the ``kpts`` argument.
        u_ql: np.ndarray
            Mass-scaled polarization vectors (in units of 1 / sqrt(amu)) of the
            phonons. Again, the ordering must be the same as in the
            corresponding ``qpts`` argument.
        omega_ql: np.ndarray
            Vibrational frequencies in eV.
        kpts_from: list[int] or int
            Calculate only the matrix element for the k-vectors specified by
            their index in the ``kpts`` argument (default: all).
        spin: int
            In case of spin-polarised system, define which spin to use
            (0 or 1).
        """
        assert len(c_kn.shape) == 3
        assert len(u_ql.shape) == 4
        if omega_ql is not None:
            assert np.all(u_ql.shape[:2] == omega_ql.shape[:2])

        # Translate k-points into 1. BZ (required by ``find_k_plus_q``` member
        # function of the ```KPointDescriptor``).
        if isinstance(kpts, np.ndarray):
            assert kpts.shape[1] == 3, "kpts_kc array must be given"
            # XXX This does not seem to cause problems!
            kpts -= kpts.round()

        # Use the KPointDescriptor to keep track of the k and q-vectors
        kd_kpts = KPointDescriptor(kpts)
        kd_qpts = KPointDescriptor(qpts)
        # Check that number of k- and q-points agree with the number of Bloch
        # functions and polarization vectors
        assert kd_kpts.nbzkpts == len(c_kn)
        assert kd_qpts.nbzkpts == len(u_ql)

        # Include all k-point per default
        if kpts_from is None:
            kpts_kc = kd_kpts.bzk_kc
            kpts_k = range(kd_kpts.nbzkpts)
        else:
            kpts_kc = kd_kpts.bzk_kc[kpts_from]
            if isinstance(kpts_from, int):
                kpts_k = list([kpts_from])
            else:
                kpts_k = list(kpts_from)

        # Number of phonon modes and electronic bands
        nmodes = u_ql.shape[1]
        nbands = c_kn.shape[1]
        # Number of atoms displacements and basis functions
        ndisp = np.prod(u_ql.shape[2:])
        assert ndisp == (3 * len(self.indices))
        nao = c_kn.shape[2]

        # Lattice vectors
        R_cN = self.compute_lattice_vectors()
        # Number of unit cell in supercell
        N = np.prod(self.supercell)

        # Allocate array for couplings
        g_qklnn = np.zeros((kd_qpts.nbzkpts, len(kpts_kc), nmodes,
                            nbands, nbands), dtype=complex)

        self.supercell_cache = MultiFileJSONCache(name)
        self.basis_info = self.supercell_cache['basis']

        parprint("Calculating coupling matrix elements")
        for q, q_c in enumerate(kd_qpts.bzk_kc):
            # Find indices of k+q for the k-points
            kplusq_k = kd_kpts.find_k_plus_q(q_c, kpts_k=kpts_k)

            # Here, ``i`` is counting from 0 and ``k`` is the global index of
            # the k-point
            for i, (k, k_c) in enumerate(zip(kpts_k, kpts_kc)):
                # Check the wave vectors (adapted to the ``KPointDescriptor``
                # class)
                kplusq_c = k_c + q_c
                kplusq_c -= kplusq_c.round()
                assert np.allclose(kplusq_c, kd_kpts.bzk_kc[kplusq_k[i]]), \
                    (i, k, k_c, q_c, kd_kpts.bzk_kc[kplusq_k[i]])

                # LCAO coefficient for Bloch states
                ck_nM = c_kn[k]
                ckplusq_nM = c_kn[kplusq_k[i]]
                # Mass scaled polarization vectors
                u_lx = u_ql[q].reshape(nmodes, ndisp)

                # Multiply phase factors
                g_lnn = np.zeros((nmodes, nbands, nbands), dtype=complex)
                for x in range(ndisp):
                    # Allocate array
                    g_MM = np.zeros((nao, nao), dtype=complex)
                    g_sNNMM = self.supercell_cache[str(x)]
                    assert nao == g_sNNMM.shape[-1]
                    for m in range(N):
                        for n in range(N):
                            phase = self._get_phase_factor(R_cN, m, n, k_c,
                                                           q_c)
                            # Sum contributions from different cells
                            g_MM += g_sNNMM[spin, m, n, :, :] * phase

                    g_nn = np.dot(ckplusq_nM.conj(), np.dot(g_MM, ck_nM.T))
                    # not sure if einsum is faster or slower
                    # g_nn = np.einsum('ij,jk,kl->il',ckplusq_nM.conj(),
                    # g_MM, ck_nM.T)
                    # g_lnn += np.outer(u_lx[:,x],g_nn).reshape(nmodes,
                    # nbands, nbands)
                    g_lnn += np.einsum('i,kl->ikl', u_lx[:, x], g_nn)

                # Insert value
                g_qklnn[q, i] = g_lnn

                # XXX Temp
                if np.all(q_c == 0.0):
                    # These should be real. Problem is... they are usually not
                    print(g_qklnn[q].imag.min(), g_qklnn[q].imag.max())

        parprint("Finished calculation of coupling matrix elements")

        # Return the bare matrix element if frequencies are not given
        if omega_ql is None:
            # Convert to eV / Ang
            g_qklnn *= units.Hartree / units.Bohr
        else:
            # Multiply prefactor sqrt(hbar / 2 * M * omega) in units of Bohr
            amu = units._amu  # atomic mass unit
            me = units._me   # electron mass
            g_qklnn /= np.sqrt(2 * amu / me / units.Hartree *
                               omega_ql[:, np.newaxis, :,
                                        np.newaxis, np.newaxis])
            # Convert to eV
            g_qklnn *= units.Hartree

        # Return couplings in eV (or eV / Ang)
        return g_qklnn

    def fourier_filter(self, V1t_xG, components='normal', criteria=0):
        """Fourier filter atomic gradients of the effective potential.

        This method is not tested.

        Parameters
        ----------
        V1t_xG: np.ndarray
            Array representation of atomic gradients of the effective potential
            in the supercell grid.
        components: str
            Fourier components to filter out (``normal`` or ``umklapp``).
        """
        import numpy.fft as fft
        import numpy.linalg as la
        assert components in ['normal', 'umklapp']
        # Grid shape
        shape = V1t_xG.shape[-3:]

        # Primitive unit cells in Bohr/Bohr^-1
        cell_cv = self.atoms.get_cell() / units.Bohr
        reci_vc = 2 * np.pi * la.inv(cell_cv)
        norm_c = np.sqrt(np.sum(reci_vc**2, axis=0))
        # Periodic BC array
        pbc_c = np.array(self.atoms.get_pbc(), dtype=bool)

        # Supercell atoms and cell
        atoms_N = self.atoms * self.supercell
        supercell_cv = atoms_N.get_cell() / units.Bohr

        # q-grid in units of the grid spacing (FFT ordering)
        q_cG = np.indices(shape).reshape(3, -1)
        q_c = np.array(shape)[:, np.newaxis]
        q_cG += q_c // 2
        q_cG %= q_c
        q_cG -= q_c // 2

        # Locate q-points inside the Brillouin zone
        if criteria == 0:
            # Works for all cases
            # Grid spacing in direction of reciprocal lattice vectors
            h_c = np.sqrt(np.sum((2 * np.pi * la.inv(supercell_cv))**2,
                                 axis=0))
            # XXX Why does a "*=" operation on q_cG not work here ??
            q1_cG = q_cG * h_c[:, np.newaxis] / (norm_c[:, np.newaxis] / 2)
            mask_G = np.ones(np.prod(shape), dtype=bool)
            for i, pbc in enumerate(pbc_c):
                if not pbc:
                    continue
                mask_G &= (-1. < q1_cG[i]) & (q1_cG[i] <= 1.)
        else:
            # 2D hexagonal lattice
            # Projection of q points onto the periodic directions. Only in
            # these directions do normal and umklapp processees make sense.
            q_vG = np.dot(q_cG[pbc_c].T,
                          2 * np.pi * la.inv(supercell_cv).T[pbc_c]).T.copy()
            # Parametrize the BZ boundary in terms of the angle theta
            theta_G = np.arctan2(q_vG[1], q_vG[0]) % (np.pi / 3)
            phi_G = np.pi / 6 - np.abs(theta_G)
            qmax_G = norm_c[0] / 2 / np.cos(phi_G)
            norm_G = np.sqrt(np.sum(q_vG**2, axis=0))
            # Includes point on BZ boundary with +1e-2
            mask_G = (norm_G <= qmax_G + 1e-2)

        if components != 'normal':
            mask_G = ~mask_G

        # Reshape to grid shape
        mask_G.shape = shape

        for V1t_G in V1t_xG:
            # Fourier transform atomic gradient
            V1tq_G = fft.fftn(V1t_G)
            # Zero normal/umklapp components
            V1tq_G[mask_G] = 0.0
            # Fourier transform back
            V1t_G[:] = fft.ifftn(V1tq_G).real

    def calculate_gradient(self):
        """Calculate gradient of effective potential and projector coefs.

        This function loads the generated pickle files and calculates
        finite-difference derivatives.

        """

        # Array and dict for finite difference derivatives
        V1t_xsG = []
        dH1_xasp = []

        x = 0
        for a in self.indices:
            for v in 'xyz':
                # Note: self.name currently ignored in ase.phonon
                # name = '%s.%d%s' % (self.name, a, v)
                name = '%d%s' % (a, v)
                # Potential and atomic density matrix for atomic displacement
                Vtm_sG = self.cache[name + '-']['Vt_sG']
                dHm_asp = self.cache[name + '-']['dH_all_asp']
                Vtp_sG = self.cache[name + '+']['Vt_sG']
                dHp_asp = self.cache[name + '+']['dH_all_asp']

                # FD derivatives in Hartree / Bohr
                V1t_sG = (Vtp_sG - Vtm_sG) / (2 * self.delta / units.Bohr)
                V1t_xsG.append(V1t_sG)

                dH1_asp = {}
                for atom in dHm_asp.keys():
                    dH1_asp[atom] = (dHp_asp[atom] - dHm_asp[atom]) / \
                                    (2 * self.delta / units.Bohr)
                dH1_xasp.append(dH1_asp)
                x += 1

        return np.array(V1t_xsG), dH1_xasp

    def _get_phase_factor(self, R_cN, m, n, k_c, q_c):
        Rm_c = R_cN[:, m]
        Rn_c = R_cN[:, n]
        phase = np.exp(2.j * np.pi * (np.dot(k_c, Rm_c - Rn_c)
                                      + np.dot(q_c, Rm_c)))
        return phase
