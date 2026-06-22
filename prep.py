# PREPRAES THE FILES TO BE READ BY MAT_GPT. RUN ON ONLY 1 PROCESSOR
import numpy as np
from pathlib import Path
from ase.io import read
from ase.parallel import paropen, world
from ase.units import Ha
from gpaw import GPAW, PW, FermiDirac, restart
from gpaw.response.frequencies import NonLinearFrequencyDescriptor
from gpaw.response.df import DielectricFunction
from gpaw.symmetry import Symmetry
from math import sqrt
import time
import warnings
import gpaw.mpi as mpi
import traceback
import gc

warnings.filterwarnings("ignore")
serial_comm = mpi.SerialCommunicator()
# MPI setup
comm = world
rank = comm.rank
size = comm.size
 
#stuff
a0 = 0.5291777721092
conv_ang_ev = 1973.269
conv_bohr_ev = 3728.94562737

#change only here
name="al"
domega0=0.08
omega2=10
omegamax=1000
#stop changing stuff



structure, calc = restart(f'{name}.gpw', txt=None, communicator=serial_comm)
cell_cv = structure.get_cell()
bcell_cv = 2 * np.pi * np.linalg.inv(cell_cv).T
id_a = structure.get_atomic_numbers()
spos_ac = structure.get_scaled_positions()

ibz_kpts = None
weights = None
kpts = None

# Get irreducible k-points and weights
ibz_kpts = np.array(calc.get_ibz_k_points())  # Irreducible k-points
weights = calc.get_k_point_weights()  # Weights of irreducible k-points
# Get full k-points
kpts = np.array(calc.get_bz_k_points())
num_iq=len(ibz_kpts)

symmetry = Symmetry(id_a,cell_cv)
symmetry.analyze(spos_ac)

# Call the map between the irreducible and reducible k-points
bzk_kc, weight_k, sym_k, time_reversal_k, bz2ibz_k, ibz2bz_k, bz2bz_ks = symmetry.reduce(kpts)

# Reads the system's symmetry operations

U_scc = []

with open('symmetries.txt','r') as f:
    lines = f.readlines()
    for i in range(0,len(lines),4):
        matrix_lines = lines[i+1:i+4]
        U_cc = np.array([list(map(int,line.split())) for line in matrix_lines])
        U_scc.append(U_cc)

U_scc = np.array(U_scc, dtype=int)

np.save("U_scc.npy", U_scc)


np.savetxt("b_vectors.dat", bcell_cv, fmt="%.8f")
freq = NonLinearFrequencyDescriptor(domega0/Ha,omega2/Ha,omegamax/Ha).omega_w * Ha
with open('w_list','w') as k:
    for item in freq:
        print(f"{item:.6f}", file=k)
        

time_reversal = True
if time_reversal :
    nsym = int(len(U_scc)/2)
else:
    nsym = len(U_scc)
    
sym_k_2 = -np.ones(len(sym_k), dtype=int)
for i in range(len(sym_k)):
    if time_reversal_k[i] == True:
        sym_k_2[i] = sym_k[i] + nsym
    else:
        sym_k_2[i] = sym_k[i]
np.save("sym_k_2.npy", sym_k_2)
        
N_vec_first = np.zeros([len(sym_k_2),3], dtype = int)
N_vec_try_first = np.zeros(3)
for i in range(len(sym_k_2)):
    ki = kpts[i]
    u_kirr = np.dot(U_scc[sym_k_2[i]],ibz_kpts[bz2ibz_k[i]])
    if np.array_equal(ki, u_kirr):
        N_vec_first[i] = [0,0,0]
    else :
        N_vec_try_first = ki - u_kirr
        if np.allclose(N_vec_try_first, np.round(N_vec_try_first)):
            N_vec_first[i] = np.round(N_vec_try_first)
np.save("N_vec_first.npy", N_vec_first)

U_scc_inv = []
for U_cc in U_scc:
    U_cc_inv = np.linalg.inv(U_cc)
    U_cc_inv_int = np.round(U_cc_inv).astype(int)
    U_scc_inv.append(U_cc_inv_int)

U_scc_inv = np.array(U_scc_inv, dtype=int)

np.save("U_scc_inv.npy", U_scc_inv)

inverse_idx = -np.ones(len(U_scc_inv), dtype=int)
for U_index, U_cc in enumerate(U_scc):
    for inv_index, U_cc_inv_int in enumerate(U_scc_inv):
        if np.array_equal(U_cc, U_cc_inv_int):
            inverse_idx[U_index] = inv_index
            break  # Stop after the first match

sym_k_2_inv = -np.ones(len(sym_k_2), dtype=int)
for i in range(len(sym_k_2)):
    sym_k_2_inv[i] = inverse_idx[sym_k_2[i]]
np.save("sym_k_2_inv.npy", sym_k_2_inv)

N_vec = np.zeros([len(sym_k_2),3], dtype = int)
N_vec_try = np.zeros(3)
for i in range(len(sym_k_2)):
    ki = kpts[i]
    u_ki = np.dot(U_scc[sym_k_2_inv[i]],ki)
    if np.array_equal(u_ki, ibz_kpts[bz2ibz_k[i]]):
        N_vec[i] = [0,0,0]
    else :
        N_vec_try = u_ki - ibz_kpts[bz2ibz_k[i]]
        if np.allclose(N_vec_try, np.round(N_vec_try)):
            N_vec[i] = np.round(N_vec_try) 
np.save("N_vec.npy", N_vec)


output_file = "qirr_rot.txt"
with open(output_file, 'w') as file:
    file.write(f"#k-index k-point kirr r-index s-index Nfirst\n")
    for i in range(len(sym_k_2)):
        ki = kpts[i]
        kirr = ibz_kpts[bz2ibz_k[i]]
        ri = sym_k_2[i]
        si = sym_k_2_inv[i]
        nfirst = N_vec_first[i]
        file.write(f"{i} {ki} {kirr} {ri} {si} {nfirst}\n")
        
if rank==0:
    print("DONE!")