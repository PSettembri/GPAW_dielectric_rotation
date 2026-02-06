from types import SimpleNamespace

from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.lfc import BasisFunctions
from gpaw.mpi import serial_comm


def create_basis(ibz,
                 nspins,
                 pbc_c,
                 grid,
                 setups,
                 dtype,
                 fracpos_ac,
                 comm=serial_comm,
                 kpt_comm=serial_comm,
                 band_comm=serial_comm):
    kd = KPointDescriptor(ibz.bz.kpt_Kc, nspins)
    kd.set_symmetry(SimpleNamespace(pbc=pbc_c),
                    ibz.symmetries.symmetry,
                    comm=comm)
    kd.set_communicator(kpt_comm)

    basis = BasisFunctions(grid._gd,
                           [setup.basis_functions_J for setup in setups],
                           kd,
                           dtype=dtype,
                           cut=True)
    basis.set_positions(fracpos_ac)
    myM = (basis.Mmax + band_comm.size - 1) // band_comm.size
    basis.set_matrix_distribution(
        min(band_comm.rank * myM, basis.Mmax),
        min((band_comm.rank + 1) * myM, basis.Mmax))
    return basis
