from types import SimpleNamespace
from math import pi

import numpy as np

from gpaw.new.builder import DFTComponentsBuilder
from gpaw.new.calculation import DFTState
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunction
from gpaw.new.lcao.eigensolver import LCAOEigensolver
from gpaw.new.lcao.hamiltonian import LCAOHamiltonian
from gpaw.new.pwfd.davidson import Davidson
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions


class PWFDDFTComponentsBuilder(DFTComponentsBuilder):
    def __init__(self, atoms, params, *, comm, qspiral=None):
        super().__init__(atoms, params, comm=comm)
        self.qspiral_v = (None if qspiral is None else
                          qspiral @ self.grid.icell * (2 * pi))

    def create_eigensolver(self, hamiltonian):
        eigsolv_params = self.params.eigensolver.copy()
        name = eigsolv_params.pop('name', 'dav')
        assert name == 'dav'
        return Davidson(
            self.nbands,
            self.wf_desc,
            self.communicators['b'],
            hamiltonian.create_preconditioner,
            converge_bands=self.params.convergence.get('bands', 'occupied'),
            **eigsolv_params)

    def read_ibz_wave_functions(self, reader):
        kpt_comm, band_comm, domain_comm = (self.communicators[x]
                                            for x in 'kbd')

        def create_wfs(spin: int, q: int, k: int, kpt_c, weight: float):
            psit_nG = SimpleNamespace(
                comm=band_comm,
                dims=(self.nbands,),
                desc=self.wf_desc.new(kpt=kpt_c),
                data=None,
                xp=np)
            wfs = PWFDWaveFunctions(
                spin=spin,
                q=q,
                k=k,
                weight=weight,
                psit_nX=psit_nG,  # type: ignore
                setups=self.setups,
                fracpos_ac=self.fracpos_ac,
                atomdist=self.atomdist,
                ncomponents=self.ncomponents,
                qspiral_v=self.qspiral_v)

            return wfs

        ibzwfs = PWFDIBZWaveFunction.create(
            ibz=self.ibz,
            nelectrons=self.nelectrons,
            ncomponents=self.ncomponents,
            create_wfs_func=create_wfs,
            kpt_comm=self.communicators['k'],
            kpt_band_comm=self.communicators['D'],
            comm=self.communicators['w'])

        # Set eigenvalues, occupations, etc..
        self.read_wavefunction_values(reader, ibzwfs)

        return ibzwfs

    def create_ibz_wave_functions(self, basis, potential, *, log):
        from gpaw.new.lcao.builder import create_lcao_ibzwfs

        if self.params.random:
            return self.create_random_ibz_wave_functions(log)

        # sl_default = self.params.parallel['sl_default']
        # sl_lcao = self.params.parallel['sl_lcao'] or sl_default

        lcaonbands = min(self.nbands,
                         basis.Mmax * (2 if self.ncomponents == 4 else 1))
        lcao_ibzwfs, _ = create_lcao_ibzwfs(
            basis,
            self.ibz, self.communicators, self.setups,
            self.fracpos_ac, self.grid, self.dtype,
            lcaonbands, self.ncomponents, self.atomdist, self.nelectrons)

        state = DFTState(lcao_ibzwfs, None, potential)
        hamiltonian = LCAOHamiltonian(basis)
        LCAOEigensolver(basis).iterate(state, hamiltonian)

        def create_wfs(spin, q, k, kpt_c, weight):
            lcaowfs = lcao_ibzwfs.wfs_qs[q][spin]
            assert lcaowfs.spin == spin

            # Convert to PW-coefs in PW-mode:
            psit_nX = self.convert_wave_functions_from_uniform_grid(
                lcaowfs.C_nM, basis, kpt_c, q)

            mylcaonbands, nao = lcaowfs.C_nM.dist.shape
            mynbands = len(psit_nX.data)
            eig_n = np.empty(self.nbands)
            eig_n[:lcaonbands] = lcaowfs._eig_n
            eig_n[lcaonbands:] = 1e10
            if mylcaonbands < mynbands:
                psit_nX[mylcaonbands:].randomize(
                    seed=self.communicators['w'].rank)

            wfs = PWFDWaveFunctions(
                psit_nX=psit_nX,
                spin=spin,
                q=q,
                k=k,
                weight=weight,
                setups=self.setups,
                fracpos_ac=self.fracpos_ac,
                atomdist=self.atomdist,
                ncomponents=self.ncomponents,
                qspiral_v=self.qspiral_v)
            wfs._eig_n = eig_n
            return wfs

        return PWFDIBZWaveFunction.create(
            ibz=self.ibz,
            nelectrons=self.nelectrons,
            ncomponents=self.ncomponents,
            create_wfs_func=create_wfs,
            kpt_comm=self.communicators['k'],
            kpt_band_comm=self.communicators['D'],
            comm=self.communicators['w'])

    def create_random_ibz_wave_functions(self, log):
        log('Initializing wave functions with random numbers')

        def create_wfs(spin, q, k, kpt_c, weight):
            desc = self.wf_desc.new(kpt=kpt_c)
            psit_nX = desc.empty(
                dims=(self.nbands,),
                comm=self.communicators['b'],
                xp=self.xp)
            psit_nX.randomize()

            wfs = PWFDWaveFunctions(
                psit_nX=psit_nX,
                spin=spin,
                q=q,
                k=k,
                weight=weight,
                setups=self.setups,
                fracpos_ac=self.fracpos_ac,
                atomdist=self.atomdist,
                ncomponents=self.ncomponents,
                qspiral_v=self.qspiral_v)

            eig_n = self.xp.empty(self.nbands)
            eig_n[:] = 1e10  # np.inf
            wfs._eig_n = eig_n
            return wfs

        return PWFDIBZWaveFunction.create(
            ibz=self.ibz,
            nelectrons=self.nelectrons,
            ncomponents=self.ncomponents,
            create_wfs_func=create_wfs,
            kpt_comm=self.communicators['k'],
            kpt_band_comm=self.communicators['D'],
            comm=self.communicators['w'])
