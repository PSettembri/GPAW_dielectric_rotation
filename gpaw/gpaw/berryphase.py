from __future__ import annotations
import json
import warnings
from os.path import exists, splitext
from pathlib import Path
from typing import Any

import numpy as np
from ase.dft.bandgap import bandgap
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset

from gpaw import GPAW
from gpaw.ibz2bz import get_overlap
from gpaw.ibz2bz import (get_overlap_coefficients,
                         get_phase_shifted_overlap_coefficients)
from gpaw.mpi import rank, serial_comm, world
from gpaw.spinorbit import soc_eigenstates
from gpaw.utilities.blas import gemmdot


def get_berry_phases(calc, spin=0, dir=0, check2d=False):
    if isinstance(calc, str):
        calc = GPAW(calc, communicator=serial_comm, txt=None)

    assert len(calc.symmetry.op_scc) == 1  # does not work with symmetry
    gap = bandgap(calc)[0]
    assert gap != 0.0

    M = np.round(calc.get_magnetic_moment())
    assert np.allclose(M, calc.get_magnetic_moment(), atol=0.05), \
        print(M, calc.get_magnetic_moment())
    nvalence = calc.wfs.setups.nvalence
    nocc_s = [int((nvalence + M) / 2), int((nvalence - M) / 2)]
    nocc = nocc_s[spin]
    if not calc.wfs.collinear:
        nocc = nvalence
    else:
        assert np.allclose(np.sum(nocc_s), nvalence)

    bands = list(range(nocc))
    kpts_kc = calc.get_bz_k_points()
    size = get_monkhorst_pack_size_and_offset(kpts_kc)[0]
    Nk = len(kpts_kc)
    wfs = calc.wfs

    dO_aii = get_overlap_coefficients(wfs)

    kd = calc.wfs.kd

    u_knR = []
    proj_k = []
    for k in range(Nk):
        ik = kd.bz2ibz_k[k]
        k_c = kd.bzk_kc[k]
        ik_c = kd.ibzk_kc[ik]
        # Since symmetry is off this should always hold
        assert np.allclose(k_c, ik_c)
        kpt = wfs.kpt_qs[ik][spin]

        # Check that all states are occupied
        assert np.all(kpt.f_n[:nocc] > 1e-6)
        N_c = wfs.gd.N_c

        ut_nR = []
        psit_nG = kpt.psit_nG
        for n in range(nocc):
            if wfs.collinear:
                ut_nR.append(wfs.pd.ifft(psit_nG[n], ik))
            else:
                ut0_R = wfs.pd.ifft(psit_nG[n][0], ik)
                ut1_R = wfs.pd.ifft(psit_nG[n][1], ik)
                # Here R includes a spinor index
                ut_nR.append([ut0_R, ut1_R])

        u_knR.append(ut_nR)
        proj_k.append(kpt.projections)

    indices_kkk = np.arange(Nk).reshape(size)
    tmp = np.concatenate([[i for i in range(3) if i != dir], [dir]])
    indices_kk = indices_kkk.transpose(tmp).reshape(-1, size[dir])

    nkperp = len(indices_kk)
    phases = []
    if check2d:
        phases2d = []
    for indices_k in indices_kk:
        M_knn = []
        for j in range(size[dir]):
            k1 = indices_k[j]
            G_c = np.array([0, 0, 0])
            if j + 1 < size[dir]:
                k2 = indices_k[j + 1]
            else:
                k2 = indices_k[0]
                G_c[dir] = 1
            u1_nR = np.array(u_knR[k1])
            u2_nR = np.array(u_knR[k2])
            k1_c = kpts_kc[k1]
            k2_c = kpts_kc[k2] + G_c

            if np.any(G_c):
                emiGr_R = np.exp(-2j * np.pi *
                                 np.dot(np.indices(N_c).T, G_c / N_c).T)
                u2_nR = u2_nR * emiGr_R

            bG_c = k2_c - k1_c

            phase_shifted_dO_aii = get_phase_shifted_overlap_coefficients(
                dO_aii, calc.spos_ac, -bG_c)
            M_nn = get_overlap(bands,
                               wfs.gd,
                               u1_nR,
                               u2_nR,
                               proj_k[k1],
                               proj_k[k2],
                               phase_shifted_dO_aii)
            M_knn.append(M_nn)
        det = np.linalg.det(M_knn)
        phases.append(np.imag(np.log(np.prod(det))))
        if check2d:
            # In the case of 2D systems we can check the
            # result
            k1 = indices_k[0]
            k1_c = kpts_kc[k1]
            G_c = [0, 0, 1]
            u1_nR = u_knR[k1]
            emiGr_R = np.exp(-2j * np.pi *
                             np.dot(np.indices(N_c).T, G_c / N_c).T)
            u2_nR = u1_nR * emiGr_R

            phase_shifted_dO_aii = get_phase_shifted_overlap_coefficients(
                dO_aii, calc.spos_ac, -bG_c)
            M_nn = get_overlap(bands,
                               calc.wfs.gd,
                               u1_nR,
                               u2_nR,
                               proj_k[k1],
                               proj_k[k1],
                               phase_shifted_dO_aii)

            phase2d = np.imag(np.log(np.linalg.det(M_nn)))
            phases2d.append(phase2d)

    # Make sure the phases are continuous
    for p in range(nkperp - 1):
        delta = phases[p] - phases[p + 1]
        phases[p + 1] += np.round(delta / (2 * np.pi)) * 2 * np.pi

    phase = np.sum(phases) / nkperp
    if check2d:
        for p in range(nkperp - 1):
            delta = phases2d[p] - phases2d[p + 1]
            phases2d[p + 1] += np.round(delta / (2 * np.pi)) * 2 * np.pi

        phase2d = np.sum(phases2d) / nkperp

        diff = abs(phase - phase2d)
        if diff > 0.01:
            msg = 'Warning wrong phase: phase={}, 2dphase={}'
            print(msg.format(phase, phase2d))

    return indices_kk, phases


def get_polarization_phase(calc,
                           name: str | None = None) -> np.ndarray:
    if isinstance(calc, (str, Path)):
        if name is None:
            name = splitext(calc)[0]
            berryname = f'{name}-berryphases.json'
        else:
            berryname = name
    else:
        assert calc.world.size == 1
        berryname = name or 'berryphases.json'

    phase_c = np.zeros((3,), float)
    if not exists(berryname) and world.rank == 0:
        # Calculate and save berry phases
        if isinstance(calc, (str, Path)):
            calc = GPAW(calc, communicator=serial_comm)
        assert len(calc.symmetry.op_scc) == 1
        nspins = calc.wfs.nspins
        data: dict[int | str, Any] = {}
        for c in [0, 1, 2]:
            data[c] = {}
            for spin in range(nspins):
                indices_kk, phases = get_berry_phases(calc, dir=c, spin=spin)
                data[c][spin] = phases

        # Ionic contribution
        Z_a = []
        for num in calc.atoms.get_atomic_numbers():
            for ida, setup in zip(calc.wfs.setups.id_a,
                                  calc.wfs.setups):
                if abs(ida[0] - num) < 1e-5:
                    break
            Z_a.append(setup.Nv)
        data['Z_a'] = Z_a
        data['spos_ac'] = calc.spos_ac.tolist()
        if not calc.wfs.collinear:
            data['non-collinear'] = True

        with open(berryname, 'w') as fd:
            json.dump(data, fd, indent=True)

    world.barrier()
    # Read data and calculate phase
    if world.rank == 0:
        print(f'Reading berryphases {berryname}')
    with open(berryname) as fd:
        data = json.load(fd)

    for c in [0, 1, 2]:
        nspins = len(data[str(c)])
        for spin in range(nspins):
            phases = data[str(c)][str(spin)]
            phase_c[c] += np.sum(phases) / len(phases)

    # We should not multiply by two below if non-collinear
    nc = 'non-collinear' in data.keys()
    phase_c = phase_c * (2 - nc) / nspins

    Z_a = data['Z_a']
    spos_ac = data['spos_ac']
    phase_c += 2 * np.pi * np.dot(Z_a, spos_ac)

    return phase_c


def parallel_transport(calc,
                       direction=0,
                       name=None,
                       scale=1.0,
                       bands=None,
                       theta=0.0,
                       phi=0.0):
    """
    Parallel transport.
    The parallel transport algorithm corresponds to the construction
    of hybrid Wannier functions localized along the Nloc direction.
    While these are not constructed explicitly one may obtain the
    Wannier Charge centers which are given by the eigenvalues of
    the Berry phase matrix (except for a factor of 2*pi) phi_km.
    In addition, one may evaluate the expectation value of spin
    on each of these states along the easy axis (z-axis for
    nonmagnetic systems), which is given by S_km.

    Output:
    phi_km, S_km (see above)
    """
    if isinstance(calc, str):
        calc = GPAW(calc, txt=None, communicator=serial_comm)

    if bands is None:
        nv = int(calc.get_number_of_electrons())
        bands = range(nv)

    cell_cv = calc.wfs.gd.cell_cv
    icell_cv = (2 * np.pi) * np.linalg.inv(cell_cv).T
    r_g = calc.wfs.gd.get_grid_point_coordinates()

    dO_aii = get_overlap_coefficients(calc.wfs)

    N_c = calc.wfs.kd.N_c
    assert 1 in np.delete(N_c, direction)
    Nkx = N_c[0]
    Nky = N_c[1]
    Nkz = N_c[2]

    Nk = Nkx * Nky * Nkz
    Nloc = N_c[direction]
    Npar = Nk // Nloc

    # Parallelization stuff
    myKsize = -(-Npar // (world.size))
    myKrange = range(rank * myKsize,
                     min((rank + 1) * myKsize, Npar))
    myKsize = len(myKrange)

    # Get array of k-point indices of the path. q index is loc direction
    kpts_kq = []
    for k in range(Npar):
        if direction == 0:
            kpts_kq.append(list(range(k, Nkx * Nky, Nky)))
        if direction == 1:
            if Nkz == 1:
                kpts_kq.append(list(range(k * Nky, (k + 1) * Nky)))
            else:
                kpts_kq.append(list(range(k, Nkz * Nky, Nkz)))
        if direction == 2:
            kpts_kq.append(list(range(k * Nloc, (k + 1) * Nloc)))

    G_c = np.array([0, 0, 0])
    G_c[direction] = 1
    G_v = np.dot(G_c, icell_cv)

    kpts_kc = calc.get_bz_k_points()

    if Nloc > 1:
        b_c = kpts_kc[kpts_kq[0][1]] - kpts_kc[kpts_kq[0][0]]
    else:
        b_c = G_c
    phase_shifted_dO_aii = get_phase_shifted_overlap_coefficients(
        dO_aii, calc.spos_ac, -b_c)

    soc_kpts = soc_eigenstates(calc,
                               scale=scale,
                               theta=theta,
                               phi=phi)

    def projections(bz_index):
        proj = soc_kpts[bz_index].projections
        new_proj = proj.new()
        new_proj.matrix.array = proj.matrix.array.copy()
        return new_proj

    def wavefunctions(bz_index):
        return soc_kpts[bz_index].wavefunctions(
            calc, periodic=True)

    phi_km = np.zeros((Npar, len(bands)), float)
    S_km = np.zeros((Npar, len(bands)), float)
    # Loop over the direction parallel components
    for k in myKrange:
        U_qmm = [np.eye(len(bands))]
        qpts_q = kpts_kq[k]
        # Loop over kpoints in the phase direction
        for q in range(Nloc - 1):
            iq1 = qpts_q[q]
            iq2 = qpts_q[q + 1]
            # print(kpts_kc[iq1], kpts_kc[iq2])
            if q == 0:
                u1_nsG = wavefunctions(iq1)
                proj1 = projections(iq1)

            u2_nsG = wavefunctions(iq2)
            proj2 = projections(iq2)

            M_mm = get_overlap(bands,
                               calc.wfs.gd,
                               u1_nsG,
                               u2_nsG,
                               proj1,
                               proj2,
                               phase_shifted_dO_aii)

            V_mm, sing_m, W_mm = np.linalg.svd(M_mm)
            U_mm = np.dot(V_mm, W_mm).conj()
            u_mysxz = np.dot(U_mm, np.swapaxes(u2_nsG[bands], 0, 3))
            u_mxsyz = np.swapaxes(u_mysxz, 1, 3)
            u_msxyz = np.swapaxes(u_mxsyz, 1, 2)
            u2_nsG[bands] = u_msxyz
            for a in range(len(calc.atoms)):
                assert not proj2.collinear
                P2_msi = proj2[a][bands]
                for s in range(2):
                    P2_mi = P2_msi[:, s]
                    P2_mi = np.dot(U_mm, P2_mi)
                    P2_msi[:, s] = P2_mi
                proj2[a][bands] = P2_msi
            U_qmm.append(U_mm)
            u1_nsG = u2_nsG
            proj1 = proj2
        U_qmm = np.array(U_qmm)

        # Fix phases for last point
        iq0 = qpts_q[0]
        if Nloc == 1:
            u1_nsG = wavefunctions(iq0)
            proj1 = projections(iq0)
        u2_nsG = wavefunctions(iq0)
        u2_nsG[:] *= np.exp(-1.0j * gemmdot(G_v, r_g, beta=0.0))
        proj2 = projections(iq0)

        M_mm = get_overlap(bands,
                           calc.wfs.gd,
                           u1_nsG,
                           u2_nsG,
                           proj1,
                           proj2,
                           phase_shifted_dO_aii)

        V_mm, sing_m, W_mm = np.linalg.svd(M_mm)
        U_mm = np.dot(V_mm, W_mm).conj()
        u_mysxz = np.dot(U_mm, np.swapaxes(u2_nsG[bands], 0, 3))
        u_mxsyz = np.swapaxes(u_mysxz, 1, 3)
        u_msxyz = np.swapaxes(u_mxsyz, 1, 2)
        u2_nsG[bands] = u_msxyz
        for a in range(len(calc.atoms)):
            assert not proj2.collinear
            P2_msi = proj2[a][bands]
            for s in range(2):
                P2_mi = P2_msi[:, s]
                P2_mi = np.dot(U_mm, P2_mi)
                P2_msi[:, s] = P2_mi
            proj2[a][bands] = P2_msi

        # Get overlap between first kpts and its smoothly translated image
        u2_nsG[:] *= np.exp(1.0j * gemmdot(G_v, r_g, beta=0.0))
        u1_nsG = wavefunctions(iq0)
        proj1 = projections(iq0)
        M_mm = get_overlap(bands,
                           calc.wfs.gd,
                           u1_nsG,
                           u2_nsG,
                           proj1,
                           proj2,
                           dO_aii)

        l_m, l_mm = np.linalg.eig(M_mm)
        phi_km[k] = np.angle(l_m)

        A_mm = np.zeros_like(l_mm, complex)
        for q in range(Nloc):
            iq = qpts_q[q]
            U_mm = U_qmm[q]
            v_mn = soc_kpts[iq].v_mn
            v_nm = np.einsum('xm, mn -> nx', U_mm, v_mn[bands])
            A_mm += np.dot(v_nm[::2].T.conj(), v_nm[::2])
            A_mm -= np.dot(v_nm[1::2].T.conj(), v_nm[1::2])
        A_mm /= Nloc
        S_km[k] = np.diag(l_mm.T.conj().dot(A_mm).dot(l_mm)).real

    world.sum(phi_km)
    world.sum(S_km)

    if not calc.density.collinear:
        warnings.warn('WARNING: Spin projections are not meaningful ' +
                      'for non-collinear calculations')

    if name is not None:
        np.savez(f'phases_{name}.npz', phi_km=phi_km, S_km=S_km)

    return phi_km, S_km
