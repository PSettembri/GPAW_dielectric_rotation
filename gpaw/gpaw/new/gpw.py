from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Union

import ase.io.ulm as ulm
import gpaw
import gpaw.mpi as mpi
import numpy as np
from ase import Atoms
from ase.io.trajectory import read_atoms, write_atoms
from ase.units import Bohr, Ha
from gpaw.core.atom_arrays import AtomArraysLayout
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.new.builder import builder as create_builder
from gpaw.new.calculation import DFTCalculation, DFTState, units
from gpaw.new.density import Density
from gpaw.new.input_parameters import InputParameters
from gpaw.new.logger import Logger
from gpaw.new.potential import Potential
from gpaw.typing import DTypeLike
from gpaw.utilities import unpack_hermitian, unpack_density

ENERGY_NAMES = ['kinetic', 'coulomb', 'zero', 'external', 'xc', 'entropy',
                'total_free', 'total_extrapolated',
                'band', 'stress', 'spinorbit']


def write_gpw(filename: str,
              atoms,
              params,
              dft: DFTCalculation,
              skip_wfs: bool = True) -> None:

    comm = dft.comm

    writer: ulm.Writer | ulm.DummyWriter
    if comm.rank == 0:
        writer = ulm.Writer(filename, tag='gpaw')
    else:
        writer = ulm.DummyWriter()

    with writer:
        writer.write(version=4,
                     gpaw_version=gpaw.__version__,
                     ha=Ha,
                     bohr=Bohr)

        write_atoms(writer.child('atoms'), atoms)

        results = {key: value * units[key]
                   for key, value in dft.results.items()}
        writer.child('results').write(**results)

        p = {k: v for k, v in params.items() if k not in ['txt', 'parallel']}
        # ULM does not know about numpy dtypes:
        if 'dtype' in p:
            p['dtype'] = np.dtype(p['dtype']).name
        writer.child('parameters').write(**p)

        state = dft.state
        state.density.write(writer.child('density'))
        state.potential._write_gpw(writer.child('hamiltonian'),
                                   dft.state.ibzwfs)
        wf_writer = writer.child('wave_functions')
        state.ibzwfs.write(wf_writer, skip_wfs)

        if not skip_wfs and params.mode['name'] == 'pw':
            write_wave_function_indices(wf_writer,
                                        state.ibzwfs,
                                        state.density.nt_sR.desc)

    comm.barrier()


def write_wave_function_indices(writer, ibzwfs, grid):
    if ibzwfs.band_comm.rank != 0:
        return
    if ibzwfs.domain_comm.rank != 0:
        return

    kpt_comm = ibzwfs.kpt_comm
    ibz = ibzwfs.ibz
    nG = ibzwfs.get_max_shape(global_shape=True)[-1]

    writer.add_array('indices', (len(ibz), nG), np.int32)

    index_G = np.zeros(nG, np.int32)
    size = tuple(grid.size)
    if ibzwfs.dtype == float:
        size = (size[0], size[1], size[2] // 2 + 1)

    for k, rank in enumerate(ibzwfs.rank_k):
        if rank == kpt_comm.rank:
            wfs = ibzwfs.wfs_qs[ibzwfs.q_k[k]][0]
            i_G = wfs.psit_nX.desc.indices(size)
            index_G[:len(i_G)] = i_G
            index_G[len(i_G):] = -1
            if rank == 0:
                writer.fill(index_G)
            else:
                kpt_comm.send(index_G, 0)
        elif kpt_comm.rank == 0:
            kpt_comm.receive(index_G, rank)
            writer.fill(index_G)


def read_gpw(filename: Union[str, Path, IO[str]],
             *,
             log: Union[Logger, str, Path, IO[str]] = None,
             comm=None,
             parallel: dict[str, Any] = None,
             dtype: DTypeLike = None) -> tuple[Atoms,
                                               DFTCalculation,
                                               InputParameters,
                                               DFTComponentsBuilder]:
    """
    Read gpw file

    Returns
    -------
    atoms, calculation, params, builder
    """
    parallel = parallel or {}

    if not isinstance(log, Logger):
        log = Logger(log, comm or mpi.world)

    comm = log.comm

    log(f'Reading from {filename}')

    reader = ulm.Reader(filename)
    bohr = reader.bohr
    ha = reader.ha

    atoms = read_atoms(reader.atoms)
    kwargs = reader.parameters.asdict()
    kwargs['parallel'] = parallel

    if 'dtype' in kwargs:
        kwargs['dtype'] = np.dtype(kwargs['dtype'])

    # kwargs['nbands'] = reader.wave_functions.eigenvalues.shape[-1]

    for old_keyword in ['fixdensity', 'txt']:
        kwargs.pop(old_keyword, None)

    params = InputParameters(kwargs, warn=False)
    builder = create_builder(atoms, params, comm)

    if comm.rank == 0:
        nt_sR_array = reader.density.density * bohr**3
        vt_sR_array = reader.hamiltonian.potential / ha
        if builder.xc.type == 'MGGA':
            taut_sR_array = reader.density.ked * (bohr**3 / ha)
            dedtaut_sR_array = reader.hamiltonian.mgga_potential * bohr**-3
        D_sap_array = reader.density.atomic_density_matrices
        dH_sap_array = reader.hamiltonian.atomic_hamiltonian_matrices / ha
        shape = nt_sR_array.shape[1:]
    else:
        nt_sR_array = None
        vt_sR_array = None
        taut_sR_array = None
        dedtaut_sR_array = None
        D_sap_array = None
        dH_sap_array = None
        shape = None

    if builder.grid.global_shape() != mpi.broadcast(shape, comm=comm):
        # old gpw-file:
        kwargs.pop('h', None)
        kwargs['gpts'] = nt_sR_array.shape[1:]
        params = InputParameters(kwargs, warn=False)
        builder = create_builder(atoms, params, comm)

    if dtype is not None:
        params.mode['dtype'] = dtype

    (kpt_comm, band_comm, domain_comm, kpt_band_comm) = (
        builder.communicators[x] for x in 'kbdD')

    nt_sR = builder.grid.empty(builder.ncomponents)
    vt_sR = builder.grid.empty(builder.ncomponents)

    if builder.xc.type == 'MGGA':
        taut_sR = builder.grid.empty(builder.ncomponents)
        dedtaut_sR = builder.grid.empty(builder.ncomponents)
    else:
        taut_sR = None
        dedtaut_sR = None

    dtype = float if builder.ncomponents < 4 else complex
    atom_array_layout = AtomArraysLayout(
        [(setup.ni * (setup.ni + 1) // 2) for setup in builder.setups],
        atomdist=builder.atomdist, dtype=dtype)
    D_asp = atom_array_layout.empty(builder.ncomponents)
    dH_asp = atom_array_layout.empty(builder.ncomponents)

    if kpt_band_comm.rank == 0:
        nt_sR.scatter_from(nt_sR_array)
        vt_sR.scatter_from(vt_sR_array)
        if builder.xc.type == 'MGGA':
            taut_sR.scatter_from(taut_sR_array)
            dedtaut_sR.scatter_from(dedtaut_sR_array)
        D_asp.scatter_from(D_sap_array)
        dH_asp.scatter_from(dH_sap_array)
    if reader.version < 4:
        convert_to_new_packing_convention(D_asp, density=True)
        convert_to_new_packing_convention(dH_asp)

    kpt_band_comm.broadcast(nt_sR.data, 0)
    kpt_band_comm.broadcast(vt_sR.data, 0)
    if builder.xc.type == 'MGGA':
        kpt_band_comm.broadcast(taut_sR.data, 0)
        kpt_band_comm.broadcast(dedtaut_sR.data, 0)
    kpt_band_comm.broadcast(D_asp.data, 0)
    kpt_band_comm.broadcast(dH_asp.data, 0)

    if reader.version >= 4:
        if comm.rank == 0:
            vHt_x_array = reader.hamiltonian.electrostatic_potential / ha
        else:
            vHt_x_array = None
        vHt_x = builder.electrostatic_potential_desc.empty()
        if kpt_band_comm.rank == 0:
            vHt_x.scatter_from(vHt_x_array)
        kpt_band_comm.broadcast(vHt_x.data, 0)
    else:
        vHt_x = None

    density = Density.from_data_and_setups(
        nt_sR, taut_sR, D_asp.to_full(),
        builder.params.charge,
        builder.setups,
        builder.get_pseudo_core_densities(),
        builder.get_pseudo_core_ked())
    energies = {name: reader.hamiltonian.get(f'e_{name}', np.nan) / ha
                for name in ENERGY_NAMES}
    penergies = {key: e for key, e in energies.items()
                 if not key.startswith('total')}
    e_band = penergies.pop('band', np.nan)
    e_entropy = penergies.pop('entropy')
    penergies['kinetic'] -= e_band

    potential = Potential(vt_sR, dH_asp.to_full(), dedtaut_sR, penergies,
                          vHt_x)

    ibzwfs = builder.read_ibz_wave_functions(reader)
    ibzwfs.energies = {
        'band': e_band,
        'entropy': e_entropy,
        'extrapolation': (energies['total_extrapolated'] -
                          energies['total_free'])}

    dft = DFTCalculation(
        DFTState(ibzwfs, density, potential),
        builder.setups,
        builder.create_scf_loop(),
        pot_calc=builder.create_potential_calculator(),
        log=log)

    results = {key: value / units[key]
               for key, value in reader.results.asdict().items()}

    if results:
        log(f'Read {", ".join(sorted(results))}')

    dft.results = results

    if builder.mode in ['pw', 'fd']:  # fd = finite-difference
        data = ibzwfs.wfs_qs[0][0].psit_nX.data
        if not hasattr(data, 'fd'):  # fd = file-descriptor
            reader.close()
    else:
        reader.close()

    return atoms, dft, params, builder


def convert_to_new_packing_convention(a_asp, density=False):
    """Convert from old to new convention.

    ::

        1 2 3      1 2 4
        . 4 5  ->  . 3 5
        . . 6      . . 6
    """
    for a_sp in a_asp.values():
        if density:
            a_sii = unpack_density(a_sp)
        else:
            a_sii = unpack_hermitian(a_sp)
        L = np.tril_indices(a_sii.shape[1])
        a_sp[:] = a_sii[(...,) + L]
