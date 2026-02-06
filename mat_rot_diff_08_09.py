import numpy as np
from pathlib import Path

from ase import Atoms
from ase.io import read
from ase.build import bulk
from ase.parallel import paropen, world
from ase.units import Ha
from ase.optimize.bfgs import BFGS
from ase.filters import UnitCellFilter,FrechetCellFilter,StrainFilter

from gpaw import GPAW, PW, FermiDirac, restart 
from gpaw.response.frequencies import NonLinearFrequencyDescriptor
from gpaw.response.df import DielectricFunction
from gpaw.symmetry import Symmetry

from math import sqrt
from scipy.interpolate import griddata
import time
import warnings

from scipy.interpolate import LinearNDInterpolator
from mpi4py import MPI

start_time = time.time()

# File name

name = "mgb2_20_20_16_600"
name_reduced = "mgb2_10_10_8_600"

metal = True
#metal = False

#reduced = True
reduced = False

tensor = True
#tensor = False

# Dielectric function parameters
etav = 0.01
ecutv = 50
nbandsv = 50

# Parameters for the frequency grid
domega0 = 0.05
omega2 = 10
omegamax = 50

##############################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Conversion parameter
a0 = 0.5291777721092
conv_ang_ev = 1973.269
conv_bohr_ev = 3728.94562737

# Frequencies written in w_list file
freq = NonLinearFrequencyDescriptor(domega0/Ha,omega2/Ha,omegamax/Ha).omega_w * Ha
with open('w_list','w') as k:
    for item in freq:
        print(f"{item:.6f}", file=k)

k.close()
nw = len(freq)

# Restart from the nscf computation

if reduced:
    structure,calc = restart(f'{name_reduced}.gpw')
else : 
    structure,calc = restart(f'{name}.gpw')

# Info on the system's cell
cell_cv = structure.get_cell()
bcell_cv = 2 * np.pi * np.linalg.inv(cell_cv).T
id_a = structure.get_atomic_numbers()
spos_ac = structure.get_scaled_positions()

np.savetxt("b_vectors.dat", bcell_cv, fmt="%.8f")

# Informatrions on k-points and symmetry operations

# Get irreducible k-points and weights
ibz_kpts = np.array(calc.get_ibz_k_points())  # Irreducible k-points
weights = calc.get_k_point_weights()  # Weights of irreducible k-points

# Get full k-points
kpts = np.array(calc.get_bz_k_points())

# Initialize symmetry class
symmetry = Symmetry(id_a,cell_cv)
symmetry.analyze(spos_ac)

#print('Number of symmetry operations:', len(symmetry.op_scc))

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

# Number of symmetry operations 
# ! switch off if the time reversal symmetry was not applied
time_reversal = True
if time_reversal :
    nsym = int(len(U_scc)/2)
else :
    nsym = len(U_scc)

# Updated ist of symmetries from qirr to q 
sym_k_2 = -np.ones(len(sym_k), dtype=int)

for i in range(len(sym_k)):
    if time_reversal_k[i] == True :
        sym_k_2[i] = sym_k[i] + nsym
    else :
         sym_k_2[i] = sym_k[i]

# Inverse of the symmetry matrices

U_scc_inv = []
for U_cc in U_scc:
    U_cc_inv = np.linalg.inv(U_cc)
    U_cc_inv_int = np.round(U_cc_inv).astype(int)   
    U_scc_inv.append(U_cc_inv_int)

# Map that gives for each symmetry the index of its inverse

inverse_idx = -np.ones(len(U_scc_inv), dtype=int)
for U_index, U_cc in enumerate(U_scc):
    for inv_index, U_cc_inv_int in enumerate(U_scc_inv):
        if np.array_equal(U_cc, U_cc_inv_int):
            inverse_idx[U_index] = inv_index
            break  # Stop after the first match

sym_k_2_inv = -np.ones(len(sym_k_2), dtype=int)
for i in range(len(sym_k_2)):
    sym_k_2_inv[i] = inverse_idx[sym_k_2[i]]

# Repeat the operations and find N vectors
# qirr = Us*k - N   s:sym_k_2_inv

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

# q = Ur*qirr + N'  r:sym_k_2
# UrN = N' 

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

###########

initialization_time = time.time()

time_loop_q_print = initialization_time

# Loop over the wave-vectors q
nq = len(kpts)
nqirr = len(ibz_kpts) 

# Initialize file 'q_list' with info on the q-vectors
f = paropen('q_list','w')
f.write(
        f"{nqirr} {nq} \n"
    )

#eps0_irr = [None] * nqirr
#eps_irr = [None] * nqirr
G_irr = [None] * nqirr
eps_q = [None] * nq
G_q = [None] * nq


# Restart from the calculation with larger grid
if reduced :
    structure,calc = restart(f'{name}.gpw')

#for i in range(nq):
for iq in range(nqirr):
#for iq in range(1):

    # Definition of the parameters for the dielectric function calculation
    if metal:
        df = DielectricFunction(calc=f'{name}.gpw',
                eta=etav,
                frequencies={'type': 'nonlinear', 'domega0': domega0, 'omega2': omega2, 'omegamax': omegamax},
                ecut=ecutv,
                nbands=nbandsv,
                rate='eta',
                txt='out_df_%d.txt' %iq)
    else:
        df = DielectricFunction(calc=f'{name}.gpw',
                eta=etav,
                frequencies={'type': 'nonlinear', 'domega0': domega0, 'omega2': omega2, 'omegamax': omegamax},
                ecut=ecutv,
                nbands=nbandsv,
                txt='out_df_%d.txt' %iq)


    # q vector is selected
    #q_c = kpts[i]
    q_c = ibz_kpts[iq]
    #q_c = [-0.25 , 0.5 , 0.5]

    # The modified GPAW function to compute eps_GG(q,w) is called
    eps0 , eps = df.get_full_dielectric_function(q_c=q_c, filename="eps_%d.csv" % iq)
   
    del eps0 

    if rank !=0 :
        del eps

    if rank == 0:

        #Time required by the calculation of eps
        if iq == 0:
            time_loop_q_calc = time.time()

    # Conversion of q to cartesian and print of relative info in q_list
        q_v = np.dot(q_c, bcell_cv)
        mod = sqrt(np.inner(q_v, q_v))*conv_ang_ev   
        f.write(
            f"{iq} "
            f"{q_c[0]:.6f} {q_c[1]:.6f} {q_c[2]:.6f} "
            f"{q_v[0]:.6f} {q_v[1]:.6f} {q_v[2]:.6f} "
            f"{mod:.6f}\n"
        )

    # The reciprocal lattice vectors used for a given q are printed in dump.txt by GPAW and then read
        g_bohr = np.loadtxt('dump.txt', delimiter=',', dtype=float, encoding='utf-8')

    # Conversion and calculation of G, q+G and relative quantities
        g_cart = g_bohr/a0
        d = np.linalg.inv(bcell_cv.T)

        mod_G = np.zeros(len(g_cart), dtype=float)
        mod_qG = np.zeros(len(g_cart), dtype=float)
        g_frac = np.zeros((len(g_cart), 3), dtype=float)
        g_frac_round = np.zeros((len(g_cart), 3), dtype=int)
        qG_frac = np.zeros((len(g_cart), 3), dtype=float)
        qG_cart = np.zeros((len(g_cart), 3), dtype=float)

        for j in range(len(g_cart)):
            o = np.array(g_cart[j])
            g_frac[j] = d @ o
            g_frac_round[j] = np.round(g_frac[j]).astype(int)
            mod_G[j] = sqrt(np.inner(g_cart[j], g_cart[j]))*conv_ang_ev  #In eV
            qG_frac[j] = q_c + g_frac_round[j]
            qG_cart[j] = q_v + g_cart[j]    #In ANG-1
            mod_qG[j] = sqrt(np.inner(qG_cart[j],qG_cart[j]))*conv_ang_ev #In eV

    #G_irr[i] = g_frac_round

    # Writing of the information on G and q+G in the eps_i.csv file

        file_path=Path('eps_%d.csv' % iq)
    
        with open(file_path, 'r') as file:
            lines = file.readlines()
    
        # The lines appropriately left empty are filled with the info on G and G+q
        empty_line_index=0

        formatted_lines = [
            f"{g_frac_round[idx][0]:d} {g_frac_round[idx][1]:d} {g_frac_round[idx][2]:d} "
            f"{g_cart[idx][0]:.6f} {g_cart[idx][1]:.6f} {g_cart[idx][2]:.6f} "
            f"{mod_G[idx]:.6f} "
            f"{qG_frac[idx][0]:.6f} {qG_frac[idx][1]:.6f} {qG_frac[idx][2]:.6f} "
            f"{qG_cart[idx][0]:.6f} {qG_cart[idx][1]:.6f} {qG_cart[idx][2]:.6f} "
            f"{mod_qG[idx]:.6f}\n"
            for idx in range(len(g_frac_round))
        ]   

        for i_line, line in enumerate(lines):
            if line.strip() == "" and empty_line_index < len(formatted_lines):
                lines[i_line] = formatted_lines[empty_line_index]
                empty_line_index += 1
        lines = [line.replace('[', '').replace(']', '').replace(',', '') for line in lines]

        # Info on the q-vector is added on top of the file 
        with file_path.open('w') as file:
            file.write(
                f"{iq} "
                f"{q_c[0]:.6f} {q_c[1]:.6f} {q_c[2]:.6f} " 
                f"{q_v[0]:.6f} {q_v[1]:.6f} {q_v[2]:.6f} "
                f"{mod:.6f}\n"
            )

            file.writelines(lines)

        if iq == 0:
            time_loop_q_print = time.time()


        # Dielectric function for all q-points from their qirr

        loop_time = time.time()

        for i in range(nq):
    
            j = bz2ibz_k[i]

            if j == iq :
                eps_q[i] = eps
    
                # Lattice vectors are rotated according to symmetry
                G_q[i] = np.zeros((len(g_frac_round), 3), dtype=int)
                for ig in range(len(g_frac_round)):
                    G_q[i][ig] = np.dot(U_scc[sym_k_2[i]], g_frac_round[ig] ) - N_vec_first[i]


                fileeps = f"eps_rot_{i}.csv"  
                with open(fileeps, "w") as fd:  
                    kpts_v = np.dot(kpts[i], bcell_cv)
                    mod = sqrt(np.inner(kpts_v, kpts_v))*conv_ang_ev
                    fd.write(
                        f"{i} {nq} {len(G_q[i])} "
                        f"{kpts[i][0]:.6f} {kpts[i][1]:.6f} {kpts[i][2]:.6f} "
                        f"{kpts_v[0]:.6f} {kpts_v[1]:.6f} {kpts_v[2]:.6f} "
                        f"{mod:.6f}\n"
                    )
                    for ig in range(len(G_q[i])):
                        G_v = np.dot(G_q[i][ig], bcell_cv)
                        mod_G = sqrt(np.inner(G_v, G_v))*conv_ang_ev
                        qG_frac = kpts[i] + G_q[i][ig]
                        qG_v = np.dot(qG_frac, bcell_cv)
                        mod_qG = sqrt(np.inner(qG_v, qG_v))*conv_ang_ev

                        fd.write(f"{ig}\n")
                        fd.write(
                            f"{G_q[i][ig][0]:d} {G_q[i][ig][1]:d} {G_q[i][ig][2]:d} "
                            f"{G_v[0]:.6f} {G_v[1]:.6f} {G_v[2]:.6f} "
                            f"{mod_G:.6f} "
                            f"{qG_frac[0]:.6f} {qG_frac[1]:.6f} {qG_frac[2]:.6f} "
                            f"{qG_v[0]:.6f} {qG_v[1]:.6f} {qG_v[2]:.6f} "
                            f"{mod_qG:.6f}\n"
                        )
                        for w, rf in zip(freq, eps_q[i][:,ig]):
                        # fd.write(f"{w:.6f} {rf0.real:.6f} {rf0.imag:.6f} {rf.real:.6f} {rf.imag:.6f}\n")
                            fd.write(f"{rf.real:.10f} {rf.imag:.10f}\n")


if rank == 0:

    rotation_time = time.time()

    npoints = 0
    for i in range(nq):
        npoints += len(G_q[i])

    qG_eV_full = np.zeros((npoints,3), dtype=float)
    #eps0_qG_full = np.zeros((npoints,nw), dtype=complex)
    eps_qG_full = np.zeros((npoints,nw), dtype=complex)
    j = 0
    with open("original_grid", "w") as fd:
        for i in range(nq):
            for ig in range(len(G_q[i])):
                qG_tmp = kpts[i] + G_q[i][ig]
                qG_eV_tmp = np.dot(qG_tmp, bcell_cv)*conv_ang_ev
                qG_eV_full[j,:] = qG_eV_tmp[:]
                fd.write(f'{qG_eV_full[j,0]:.6f} {qG_eV_full[j,1]:.6f} {qG_eV_full[j,2]:.6f} \n')

                #eps0_qG_full[j,:] = eps0_q[i][:,ig]
                eps_qG_full[j,:] = eps_q[i][:,ig]

                if np.all(np.isclose(qG_eV_full[j,:] , 0, atol=1e-8)):
                    jstar = j

                j += 1
            
# eps00(0,w) assigned before interpolation
    with open("eps_gamma.dat", "w") as fd:
        fd.write(f'{qG_eV_full[jstar,0]:.6f} {qG_eV_full[jstar,1]:.6f} {qG_eV_full[jstar,2]:.6f} \n')
        for iw in range(nw):
            #fd.write(f'{freq[iw]:.6f} {eps0_qG_full[jstar,iw].real:.6f} {eps0_qG_full[jstar,iw].imag:.6f} {eps_qG_full[jstar,iw].real:.6f} {eps_qG_full[jstar,iw].imag:.6f} \n')
            fd.write(f'{freq[iw]:.6f} {eps_qG_full[jstar,iw].real:.6f} {eps_qG_full[jstar,iw].imag:.6f} \n') 

    
if tensor : 
    if metal:
        df = DielectricFunction(calc=f'{name}.gpw',
                eta=etav,
                frequencies={'type': 'nonlinear', 'domega0': domega0, 'omega2': omega2, 'omegamax': omegamax},
                ecut=ecutv,
                nbands=nbandsv,
                rate='eta',
                txt='out_df_gamma_tensor.txt')
    else:
        df = DielectricFunction(calc=f'{name}.gpw',
                eta=etav,
                frequencies={'type': 'nonlinear', 'domega0': domega0, 'omega2': omega2, 'omegamax': omegamax},
                ecut=ecutv,
                nbands=nbandsv,
                txt='out_df_gamma_tensor.txt')
        
    for dir in ['x','y','z']:
        q_c = [0.0, 0.0 , 0.0]

        g_eps0 , g_eps = df.get_dielectric_function(q_c=q_c, direction=dir)

        with open(f"eps_G_{dir}{dir}.dat", "w") as fd:
            for iw in range(nw):
                fd.write(f'{g_eps[iw].real:.6f} {g_eps[iw].imag:.6f} \n')


    end_time = time.time()
    f = paropen('times.txt','w')
    print(f"Initialization time: {initialization_time - start_time:.5f} seconds",file=f)
    print(f"Calc time first q : {time_loop_q_calc - initialization_time:.5f} seconds",file=f)
    print(f"Print time first q : {time_loop_q_print - time_loop_q_calc:.5f} seconds",file=f)
    print(f"Time spent in loop: {loop_time - initialization_time:.5f} seconds",file=f)
    print(f"Rotation time: {rotation_time - loop_time:.5f} seconds",file=f)
    print(f"Total time: {end_time - start_time:.5f} seconds",file=f)
    f.close()
