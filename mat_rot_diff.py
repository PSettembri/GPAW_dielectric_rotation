# mat_diff.py (OPTIMIZED + MEMORY SAFE)

import numpy as np
from pathlib import Path
from ase.parallel import world
from ase.units import Ha
from gpaw import restart
from gpaw.response.frequencies import NonLinearFrequencyDescriptor
from gpaw.response.df import DielectricFunction
from gpaw.symmetry import Symmetry
from math import sqrt
import warnings
import gpaw.mpi as mpi
import traceback
import gc

warnings.filterwarnings("ignore")

serial_comm = mpi.SerialCommunicator()
comm = world
rank = comm.rank
size = comm.size

def calculate_df(name, metal=True, nbandsv=50, ecutv=75, etav=0.1, domega0=0.01, omega2=0.3, omegamax=6.0):

    a0 = 0.5291777721092
    conv_ang_ev = 1973.269

    structure, calc = restart(f'{name}.gpw', txt=None, communicator=serial_comm)

    cell_cv = structure.get_cell()
    bcell_cv = 2 * np.pi * np.linalg.inv(cell_cv).T

    id_a = structure.get_atomic_numbers()
    spos_ac = structure.get_scaled_positions()

    ibz_kpts = np.array(calc.get_ibz_k_points())
    kpts = np.array(calc.get_bz_k_points())

    num_iq = len(ibz_kpts)

    symmetry = Symmetry(id_a, cell_cv)
    symmetry.analyze(spos_ac)

    bzk_kc, weight_k, sym_k, time_reversal_k, bz2ibz_k, ibz2bz_k, bz2bz_ks = symmetry.reduce(kpts)

    U_scc = np.load("U_scc.npy")
    sym_k_2 = np.load("sym_k_2.npy")
    N_vec_first = np.load("N_vec_first.npy")

    freq = NonLinearFrequencyDescriptor(domega0/Ha, omega2/Ha, omegamax/Ha).omega_w * Ha
    nw = len(freq)

    nq = len(kpts)
    nqirr = len(ibz_kpts)

    G_q = [None] * nq

    iq_indices_for_this_rank = [iq for iq in range(num_iq) if iq % size == rank]
    count = 1
    for iq in iq_indices_for_this_rank:

        if Path(f"done_{iq}.flag").exists():
            print(f"[Rank {rank}] Skipping iq={iq}", flush=True)
            continue

        try:
            q_c = ibz_kpts[iq]

            df = DielectricFunction(
                calc=f'{name}.gpw',
                eta=etav,
                frequencies={'type': 'nonlinear', 'domega0': domega0,'omega2': omega2, 'omegamax': omegamax},
                ecut=ecutv,
                nbands=nbandsv,
                rate='eta' if metal else None,
                txt=f'out_df_{iq}.txt',
                world=serial_comm
            )
            
            print(f"[Rank {rank}] Starting eps for iq={iq}, {count} out of {len(iq_indices_for_this_rank)} ({num_iq} total)", flush=True)
            eps0, eps = df.get_full_dielectric_function(
                q_c=q_c,
                filename=f"eps_{iq}.csv",
                dump_name=f"dump_{iq}.txt",
                write_all=True
            )

            # SAVE ONLY (no duplication)
            for i in range(nq):
                if bz2ibz_k[i] == iq:
                    np.save(f'eps_q{i}.npy', eps)

            q_v = np.dot(q_c, bcell_cv)
            mod = sqrt(np.inner(q_v, q_v)) * conv_ang_ev

            with open(f'q_list_{iq}.tmp', 'a') as f_rank:
                f_rank.write(
                        f"{iq} "
                        f"{q_c[0]:.6f} {q_c[1]:.6f} {q_c[2]:.6f} "
                        f"{q_v[0]:.6f} {q_v[1]:.6f} {q_v[2]:.6f} "
                        f"{mod:.6f}\n"
                        )

            # VECTORIZE G COMPUTATION
            g_bohr = np.loadtxt(f'dump_{iq}.txt', delimiter=',', dtype=float)
            g_cart = g_bohr / a0

            d = np.linalg.inv(bcell_cv.T)
            g_frac = (d @ g_cart.T).T
            g_frac_round = np.round(g_frac).astype(int)

            # WRITE ROTATED FILE DIRECTLY
            for i in range(nq):
                if bz2ibz_k[i] != iq:
                    continue
                G_q_local = np.dot(g_frac_round, U_scc[sym_k_2[i]].T) - N_vec_first[i]

                with open(f"eps_rot_{i}.csv", "w") as fd:
                    kpts_v = np.dot(kpts[i], bcell_cv)
                    mod = sqrt(np.inner(kpts_v, kpts_v)) * conv_ang_ev

                    fd.write(f"{i} {nq} {len(G_q_local)} "
                             f"{kpts[i][0]:.6f} {kpts[i][1]:.6f} {kpts[i][2]:.6f} "
                             f"{kpts_v[0]:.6f} {kpts_v[1]:.6f} {kpts_v[2]:.6f} "
                             f"{mod:.6f}\n")

                    for ig in range(len(G_q_local)):
                        G_v = np.dot(G_q_local[ig], bcell_cv)
                        qG_frac = kpts[i] + G_q_local[ig]
                        qG_v = np.dot(qG_frac, bcell_cv)

                        fd.write(f"{ig}\n")

                        fd.write(
                            f"{G_q_local[ig][0]} {G_q_local[ig][1]} {G_q_local[ig][2]} "
                            f"{G_v[0]:.6f} {G_v[1]:.6f} {G_v[2]:.6f} "
                            f"{sqrt(np.inner(G_v, G_v))*conv_ang_ev:.6f} "
                            f"{qG_frac[0]:.6f} {qG_frac[1]:.6f} {qG_frac[2]:.6f} "
                            f"{qG_v[0]:.6f} {qG_v[1]:.6f} {qG_v[2]:.6f} "
                            f"{sqrt(np.inner(qG_v, qG_v))*conv_ang_ev:.6f}\n"
                        )

                        for iw in range(nw):
                            val = eps[iw, ig]
                            fd.write(f"{val.real:.10f} {val.imag:.10f}\n")

            # FREE MEMORY IMMEDIATELY
            del eps
            del eps0
            gc.collect()
            print(f"[Rank {rank}] Completed iq={iq} out of {len(iq_indices_for_this_rank)} ({num_iq} total)", flush=True)
        except Exception as e:
            print(f"[Rank {rank}] Error at iq={iq}: {e}", flush=True)
            traceback.print_exc()
            continue

        Path(f"done_{iq}.flag").touch()
        count+=1

    world.barrier()

    if rank == 0:

        print("Redefining G_q", flush=True)

        # --- count total points ---
        npoints = 0
        for i in range(nq):
            iq = bz2ibz_k[i]
            g = np.loadtxt(f'dump_{iq}.txt', delimiter=',')
            npoints += len(g)

        # --- memory-safe arrays ---
        eps_qG_full = np.memmap(
            'eps_qG_full.dat',
            dtype='complex128',
            mode='w+',
            shape=(npoints, nw)
        )

        qG_eV_full = np.zeros((npoints, 3))

        j = 0
        jstar = 0

        d = np.linalg.inv(bcell_cv.T)
        print("creating original_grid")
        with open("original_grid", "w") as fd:

            for i in range(nq):

                iq = bz2ibz_k[i]

                # load eps safely
                eps_i = np.load(f'eps_q{i}.npy', mmap_mode='r')

                # reconstruct G properly
                g_bohr = np.loadtxt(f'dump_{iq}.txt', delimiter=',')
                g_cart = g_bohr / a0

                g_frac = (d @ g_cart.T).T
                g_frac_round = np.round(g_frac).astype(int)

                # symmetry rotation (CRITICAL)
                G_q_local = np.dot(
                    g_frac_round,
                    U_scc[sym_k_2[i]].T
                ) - N_vec_first[i]

                for ig in range(len(G_q_local)):

                    # correct q+G construction
                    qG_tmp = kpts[i] + G_q_local[ig]
                    qG_eV_tmp = np.dot(qG_tmp, bcell_cv) * conv_ang_ev

                    qG_eV_full[j] = qG_eV_tmp

                    fd.write(
                        f"{qG_eV_tmp[0]:.6f} "
                        f"{qG_eV_tmp[1]:.6f} "
                        f"{qG_eV_tmp[2]:.6f}\n"
                    )

                    eps_qG_full[j, :] = eps_i[:, ig]

                    if np.allclose(qG_eV_tmp, 0, atol=1e-8):
                        jstar = j

                    j += 1

                del eps_i

        # --- gamma file ---
        print("creating eps_gamma")
        with open("eps_gamma.dat", "w") as fd:
            fd.write(f'{qG_eV_full[jstar,0]:.6f} {qG_eV_full[jstar,1]:.6f} {qG_eV_full[jstar,2]:.6f} \n')
            for iw in range(nw):
                val = eps_qG_full[jstar, iw]
                fd.write(
                    f"{freq[iw]:.6f} "
                    f"{val.real:.6f} "
                    f"{val.imag:.6f}\n"
                )
        print("All ranks completed.", flush=True)
        with open('q_list', 'w') as outfile:
            outfile.write(f"{nqirr} {nq} \n")  # Header
            for iq in range(nqirr):  # assuming num_iq is defined globally
                temp_filename = f'q_list_{iq}.tmp'
                if Path(temp_filename).exists():
                    with open(temp_filename, 'r') as infile:
                        outfile.writelines(infile.readlines())
                    Path(temp_filename).unlink()


if __name__ == '__main__':
    calculate_df(
        name='alu400',
        metal=True,
        nbandsv=120,
        ecutv=100,
        etav=0.2,
        domega0=0.08,
        omega2=10,
        omegamax=1100
    )

    world.barrier()

    if rank == 0:
        print("Job done!")
        q_list_file = Path("q_list")
        if q_list_file.exists():
            with open(q_list_file, "r") as f:
                first_line = f.readline()
                num_iq = int(first_line.strip().split()[0])
        # Delete unnecessary files after the job finishes
        for iq in range(num_iq):
            tmp_file = Path(f'dump_{iq}.txt')
            if tmp_file.exists():
                tmp_file.unlink()
            npy_file = Path(f'eps_q{iq}.npy')
            if npy_file.exists():
                npy_file.unlink()
            done_file = Path(f'done_{iq}.flag')
            if done_file.exists():
                done_file.unlink()