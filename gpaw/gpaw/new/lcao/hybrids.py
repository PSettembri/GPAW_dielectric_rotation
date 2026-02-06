from __future__ import annotations
from gpaw.new.lcao.eigensolver import LCAOEigensolver


class HybridXCFunctional:
    setup_name = 'PBE'

    def __init__(self, params: dict):
        name = params['name']
        print(name)

    def calculate(self, nt_sr, vxct_sr):
        # semi-local stuff here?
        ...
        return 42.0

    def calculate_paw_correction(self, setup, D_sp, dH_sp):
        ...
        return 117.0


class HybridLCAOEigensolver(LCAOEigensolver):
    def __init__(self, basis, fracpos_ac, cell_cv):
        super().__init__(basis)
        print(fracpos_ac, cell_cv)

    def iterate(self, state, hamiltonian) -> float:
        for wfs in state.ibzwfs:
            rho_MM = wfs.calculate_density_matrix()
            print(rho_MM)
        return -1.0
