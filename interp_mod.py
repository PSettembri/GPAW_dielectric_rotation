import numpy as np

from ase import Atoms
from ase.io import read
from ase.build import bulk
from ase.parallel import paropen, world
from ase.units import Ha
from ase.optimize.bfgs import BFGS
from ase.filters import UnitCellFilter,FrechetCellFilter,StrainFilter

from math import sqrt
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
import time
import random
import sys

from mpi4py import MPI
from datetime import datetime

start_time = time.time()

# Task to perform

interpolation_Montecarlo_Fibonacci_parallel = True
#interpolation_Montecarlo_Fibonacci_parallel = False

#interpolation_line_parallel = True
interpolation_line_parallel = False

##############################

#scissor_correction = True
scissor_correction = False
s = 0.0

tensor = True
#tensor = False

##############################

# Conversion parameter
a0 = 0.5291777721092
conv_ang_ev = 1973.269

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

###############################

# Frequencies read from w_list file
with open("w_list", "r") as f:
    freq = [float(line.strip()) for line in f]

nw = len(freq)

# Loop over the wave-vectors q

with open("q_list", "r") as f:
    first_line = f.readline().strip()  
    nqirr, nq = map(int, first_line.split())

if rank == 0:
    now = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
    print (f"{now} Reading...", flush=True)

#eps0_q = [None] * nq
eps_q = [None] * nq
ng = np.zeros(nq,dtype=int)

for i in range(nq):
    fileeps = f"eps_rot_{i}.csv"
    with open(fileeps, "r") as fd:
        first_line = fd.readline().strip().split()
        ng[i] = int(first_line[2])  

        #eps0_q[i] = np.zeros((nw, ng[i]), dtype=complex)
        eps_q[i] = np.zeros((nw, ng[i]), dtype=complex)

        for ig in range(ng[i]):
            fd.readline()  
            fd.readline()
            for iw in range(nw):
                data_line = list(map(float, fd.readline().strip().split()))
                #eps0_q[i][iw, ig] = complex(data_line[1], data_line[2]) 
                #eps_q[i][iw, ig] = complex(data_line[3], data_line[4])
                eps_q[i][iw, ig] = complex(data_line[0], data_line[1])


with open("original_grid", "r") as fd:
    qG_eV_full = np.array([list(map(float, line.split())) for line in fd])

npoints = len(qG_eV_full)
mod_k = np.zeros(npoints, dtype=float)
for i in range(npoints):
    mod_k[i] = np.sqrt(np.dot(qG_eV_full[i,:],qG_eV_full[i,:]))
    if (mod_k[i] == 0.0):
        g_index = i

#eps0_qG_full = np.zeros((npoints,nw), dtype=complex)
eps_qG_full = np.zeros((npoints,nw), dtype=complex)
j = 0
for i in range(nq):
    for ig in range(ng[i]):
        #eps0_qG_full[j,:] = eps0_q[i][:,ig]
        eps_qG_full[j,:] = eps_q[i][:,ig]
        j += 1

if tensor:

    eps_g_xx = np.zeros(nw, dtype=complex)
    eps_g_yy = np.zeros(nw, dtype=complex)
    eps_g_zz = np.zeros(nw, dtype=complex)
    eps_g_avg = np.zeros(nw, dtype=complex)

    with open("eps_G_xx.dat", "r") as fx:
        for iw in range(nw):
            data_line = list(map(float, fx.readline().strip().split()))
            eps_g_xx[iw] = complex(data_line[0], data_line[1])
    with open("eps_G_yy.dat", "r") as fy:
        for iw in range(nw):
            data_line = list(map(float, fy.readline().strip().split()))
            eps_g_yy[iw] = complex(data_line[0], data_line[1])
    with open("eps_G_zz.dat", "r") as fz:
        for iw in range(nw):
            data_line = list(map(float, fz.readline().strip().split()))
            eps_g_zz[iw] = complex(data_line[0], data_line[1])

    for iw in range(nw):
        eps_g_avg[iw]=(eps_g_xx[iw]+eps_g_yy[iw]+eps_g_zz[iw])/3.0

if (scissor_correction): 

    w_shifted = np.empty(nw+1, dtype=np.float64)
    w_shifted[0] = 0.0
    for iw_new in range(1,nw+1):
        w_shifted[iw_new] = freq[iw_new-1] + s
    filefreqs = "w_shifted_list"
    with open(filefreqs, "w") as ffreqs:
        for iw_new in range(nw+1):
            ffreqs.write(f'{w_shifted[iw_new]:.6f} \n')


initialization_time = time.time()

if rank == 0:
    now = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
    print (f"{now} Interpolation start", flush=True)

# Parallelized interpolation using Fibonacci sphere

if interpolation_Montecarlo_Fibonacci_parallel :

    # Parameters
    # Radius in eV
    r_min = 0.01
    #r_max = 21500
    r_max = np.max(mod_k) - 500
    num_r = 250
    # Linear spacing
    r = np.linspace(r_min,r_max,num_r)
    # Logarithmic spacing
    #r = np.logspace(np.log10(r_min), np.log10(r_max), num_r)

    # Number of points on the sphere
    goldenRatio = (1+np.sqrt(5.))/2.

    # A dense Fibonacci sphere from which angles are extracted
    n_mc = 10000
    idx_mc = np.arange(0,n_mc)
    theta_mc = np.arccos(1-2*(idx_mc+0.5)/n_mc)
    phi_mc = 2*np.pi*idx_mc/goldenRatio

    num_a_min = 10
    num_a_max = 100
    num_angles = np.zeros(num_r, dtype=int)

    if (n_mc < num_a_max):
        print ("Error n_mc too low")
        sys.exit(1)

    theta_list = []
    phi_list = []

    for ir in range(num_r):
        num_angles[ir] = int(np.round(num_a_min + (num_a_max-num_a_min)*((r[ir]-r_min)/(r_max-r_min))**2))
        theta=np.zeros(num_angles[ir])
        phi=np.zeros(num_angles[ir])
        idx_list=[]
        assigned=0
        while (assigned < num_angles[ir]):
            idx = random.randint(0,n_mc-1)
            if idx not in idx_list:
                idx_list.append(idx)
                theta[assigned]=theta_mc[idx]
                phi[assigned]=phi_mc[idx]
                assigned += 1
        
        theta_list.append(theta)
        phi_list.append(phi)

    inter_points = []

    for ir in range(num_r):
        theta = theta_list[ir]
        phi = phi_list[ir]
        num_ang = num_angles[ir]

        for iangle in range(num_ang):
            x = r[ir] * np.sin(theta[iangle]) * np.cos(phi[iangle])
            y = r[ir] * np.sin(theta[iangle]) * np.sin(phi[iangle])
            z = r[ir] * np.cos(theta[iangle])

            inter_points.append([x,y,z])

    inter_points = np.array(inter_points)

    with open("interpolation_grid_Montecarlo", "w") as fd:
        for j in range(len(inter_points)):
            fd.write(f'{inter_points[j,0]:.6f} {inter_points[j,1]:.6f} {inter_points[j,2]:.6f} \n')
    
    if tensor:
        for iw in range(nw):
            eps_qG_full[g_index,iw] = eps_g_avg[iw]

    # Original data points
    points = qG_eV_full[:]

    #values0 = eps0_qG_full[:,:]
    values = eps_qG_full[:,:]

    interpolator_real = LinearNDInterpolator(points, np.real(values[:, 0]))
    interpolator_imag = LinearNDInterpolator(points, np.imag(values[:, 0]))

    # Parallelization in w

    nw_local = nw // size
    start = rank * nw_local
    end = (rank + 1) * nw_local if rank != size - 1 else nw

    #local_eps0_inter = np.empty((end-start, len(inter_points)), dtype=complex)
    local_eps_inter = np.empty((end-start, len(inter_points)), dtype=complex)

    for iw, global_iw in enumerate(range(start, end)):

        #values0_iw = values0[:, global_iw]

        #real_part0 = np.ascontiguousarray(np.real(values0_iw)).reshape(-1,1)
        #imag_part0 = np.ascontiguousarray(np.imag(values0_iw)).reshape(-1,1)

        #interpolator_real.values = real_part0
        #interpolator_imag.values = imag_part0

        #inter_real0 = interpolator_real(inter_points)
        #inter_imag0 = interpolator_imag(inter_points)

        #local_eps0_inter[iw, :] = inter_real0 + 1j * inter_imag0

        values_iw = values[:, global_iw]

        real_part = np.ascontiguousarray(np.real(values_iw)).reshape(-1,1)
        imag_part = np.ascontiguousarray(np.imag(values_iw)).reshape(-1,1)

        interpolator_real.values = real_part
        interpolator_imag.values = imag_part

        inter_real = interpolator_real(inter_points)
        inter_imag = interpolator_imag(inter_points)

        local_eps_inter[iw, :] = inter_real + 1j * inter_imag

        if rank == 0 :
            now = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
            if iw == int(0.25*nw_local):
                print (f"{now} - 25% completed", flush=True)
            elif iw == int(0.5*nw_local):
                print (f"{now} - 50% completed", flush=True)
            elif iw == int(0.75*nw_local):
                print (f"{now} - 75% completed", flush=True)


    #gathered_eps0_inter = comm.gather(local_eps0_inter, root=0)
    gathered_eps_inter = comm.gather(local_eps_inter, root=0)

    if rank == 0:
        #eps0_inter = np.vstack(gathered_eps0_inter)
        eps_inter = np.vstack(gathered_eps_inter)

        #eps0_inter_avg = np.zeros((num_r,nw), dtype=complex)
        eps_inter_avg = np.zeros((num_r,nw), dtype=complex)

        point_index = 0

        for ir in range(num_r):
            fileint = f"eps_inter_{ir}_Montecarlo.dat"
            with open(fileint, "w") as fint:
                fint.write(f'{r[ir]:.6f} \n')

                theta = theta_list[ir]
                phi = phi_list[ir]
                num_ang = num_angles[ir]

                for iangle in range(num_ang):
                    fint.write(f'{theta[iangle]:.6f} {phi[iangle] % (2*np.pi):.6f}\n')

                    j = point_index

                    for iw in range(nw):
                        #fint.write(f'{freq[iw]:.6f} {eps0_inter[iw,j].real:.6f} {eps0_inter[iw,j].imag:.6f} {eps_inter[iw,j].real:.6f} {eps_inter[iw,j].imag:.6f} \n')
                        fint.write(f'{eps_inter[iw,j].real:.10f} {eps_inter[iw,j].imag:.10f} \n')
                        #eps0_inter_avg[ir,iw] += eps0_inter[iw,j]
                        eps_inter_avg[ir,iw] += eps_inter[iw,j]

                    point_index += 1


        for ir in range(num_r):
            #eps0_inter_avg[ir,:] /= num_angles[ir]
            eps_inter_avg[ir,:] /= num_angles[ir]

        fileavg = f"eps_inter_avg_Montecarlo.dat"
        with open(fileavg, "w") as favg:
            for ir in range(num_r):
                favg.write(f'{r[ir]:.6f} \n')
                for iw in range(nw):
                    #favg.write(f'{freq[iw]:.6f} {eps0_inter_avg[ir,iw].real:.6f} {eps0_inter_avg[ir,iw].imag:.6f} {eps_inter_avg[ir,iw].real:.6f} {eps_inter_avg[ir,iw].imag:.6f} \n')
                    favg.write(f'{freq[iw]:.6f} {eps_inter_avg[ir,iw].real:.10f} {eps_inter_avg[ir,iw].imag:.10f} \n')


        filavgdelf = f"eps_inter_avg_Montecarlo_darkelf.dat"
        with open(filavgdelf, "w") as felf:
            felf.write('w(eV) q(eV) Re_eps Im_eps \n')
            for iw in range(nw):
                felf.write(f'{freq[iw]:.6f} 0.000000 {eps_qG_full[g_index,iw].real:.10f} {eps_qG_full[g_index,iw].imag:.10f} \n')
            for ir in range(num_r):
                for iw in range(nw):
                    felf.write(f'{freq[iw]:.6f} {r[ir]:.6f} {eps_inter_avg[ir,iw].real:.10f} {eps_inter_avg[ir,iw].imag:.10f} \n')

        
        if (scissor_correction):

            print (f"Scissor start", flush=True)

            eps_inter_avg_shift = np.zeros((num_r,nw+1), dtype=complex)

            point_index = 0
            for ir in range(num_r):
                fileint = f"eps_inter_{ir}_Montecarlo_shifted.dat"
                with open(fileint, "w") as fint:
                    fint.write(f'{r[ir]:.6f} \n')
                    theta = theta_list[ir]
                    phi = phi_list[ir]
                    num_ang = num_angles[ir]

                    for iangle in range(num_ang):
                        fint.write(f'{theta[iangle]:.6f} {phi[iangle] % (2*np.pi):.6f}\n')
                        j = point_index

                        for iw_new in range(nw+1):
                            if iw_new < 2 :
                                val = eps_inter[0,j]
                            else :
                                val = eps_inter[iw_new-1,j]
                            fint.write(f'{val.real:.10f} {val.imag:.10f} \n')
                            eps_inter_avg_shift[ir,iw_new] += val
                        point_index += 1
        
            for ir in range(num_r):
                eps_inter_avg_shift[ir,:] /= num_angles[ir]

            fileavg = f"eps_inter_avg_Montecarlo_shifted.dat"
            with open(fileavg, "w") as favg:
                for ir in range(num_r):
                    favg.write(f'{r[ir]:.6f} \n')
                    for iw_new in range(nw+1):
                        favg.write(f'{w_shifted[iw_new]:.6f} {eps_inter_avg_shift[ir,iw_new].real:.10f} {eps_inter_avg_shift[ir,iw_new].imag:.10f} \n')


            filavgdelf = f"eps_inter_avg_Montecarlo_darkelf_shifted.dat"
            with open(filavgdelf, "w") as felf:
                felf.write('w(eV) q(eV) Re_eps Im_eps \n')
                felf.write(f'{w_shifted[0]:.6f} 0.000000 {eps_qG_full[g_index,0].real:.10f} {eps_qG_full[g_index,0].imag:.10f} \n')
                for iw_new in range(1,nw+1):
                    felf.write(f'{w_shifted[iw_new]:.6f} 0.000000 {eps_qG_full[g_index,iw_new-1].real:.10f} {eps_qG_full[g_index,iw_new-1].imag:.10f} \n')
                for ir in range(num_r):
                    for iw_new in range(nw+1):
                        felf.write(f'{w_shifted[iw_new]:.6f} {r[ir]:.6f} {eps_inter_avg_shift[ir,iw_new].real:.10f} {eps_inter_avg_shift[ir,iw_new].imag:.10f} \n')


# Parallelized interpolation on a line 

if interpolation_line_parallel :

    # Parameters
    # Radius in eV
    r_min = 0.001
    #r_max = 21500
    r_max = np.max(mod_k) - 200
    num_r = 500
    # Linear spacing
    r = np.linspace(r_min,r_max,num_r)
    # Logarithmic spacing
    #r = np.logspace(np.log10(r_min), np.log10(r_max), num_r)

    # Line direction is in cartesian 
    line_cart = [0.866025,-0.5,0]
    line_direction = np.array(line_cart,dtype=float)
    line_direction = line_direction/(np.linalg.norm(line_direction))
    

    # Line direction in reciprocal
    #frac_direction = [0,1,1] 
    #bcell_cv = np.loadtxt("b_vectors.dat")
    #b = bcell_cv.T 
    #line_direction = b @ frac_direction
    #line_direction = line_direction/(np.linalg.norm(line_direction))
    #direction_nonzero = line_direction[line_direction != 0]
    #direction_min_nonzero = np.min(np.abs(direction_nonzero))
    #line_cart = np.round(line_direction/direction_min_nonzero).astype(int)

    inter_points = []

    for ir in range(num_r):
            x = r[ir] * line_direction[0]
            y = r[ir] * line_direction[1]
            z = r[ir] * line_direction[2]

            inter_points.append([x,y,z])

    inter_points = np.array(inter_points)

    with open("interpolation_grid_Fibonacci", "w") as fd:
        for j in range(len(inter_points)):
            fd.write(f'{inter_points[j,0]:.6f} {inter_points[j,1]:.6f} {inter_points[j,2]:.6f} \n')

    if tensor:
        eps_g_proj = np.zeros(nw, dtype=complex)
        for iw in range(nw):
            eps_g_proj[iw]=eps_g_xx[iw]*(line_direction[0]**2)+eps_g_yy[iw]*(line_direction[1]**2)+eps_g_zz[iw]*(line_direction[2]**2)
            eps_qG_full[g_index,iw] = eps_g_proj[iw]

    # Original data points

    points = qG_eV_full[:]

    values = eps_qG_full[:,:]

    interpolator_real = LinearNDInterpolator(points, np.real(values[:, 0]))
    interpolator_imag = LinearNDInterpolator(points, np.imag(values[:, 0]))

    # Parallelization in w

    nw_local = nw // size
    start = rank * nw_local
    end = (rank + 1) * nw_local if rank != size - 1 else nw

    local_eps_inter = np.empty((end-start, len(inter_points)), dtype=complex)

    for iw, global_iw in enumerate(range(start, end)):

        values_iw = values[:, global_iw]

        real_part = np.ascontiguousarray(np.real(values_iw)).reshape(-1,1)
        imag_part = np.ascontiguousarray(np.imag(values_iw)).reshape(-1,1)

        interpolator_real.values = real_part
        interpolator_imag.values = imag_part

        inter_real = interpolator_real(inter_points)
        inter_imag = interpolator_imag(inter_points)

        local_eps_inter[iw, :] = inter_real + 1j * inter_imag

        if rank == 0 :
            now = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
            if iw == int(0.25*nw_local):
                print (f"{now} - 25% completed", flush=True)
            elif iw == int(0.5*nw_local):
                print (f"{now} - 50% completed", flush=True)
            elif iw == int(0.75*nw_local):
                print (f"{now} - 75% completed", flush=True)

    gathered_eps_inter = comm.gather(local_eps_inter, root=0)

    if rank == 0:
        eps_inter = np.vstack(gathered_eps_inter)

        fileint = f"eps_inter_{line_cart[0]}_{line_cart[1]}_{line_cart[2]}.dat"
        with open(fileint, "w") as fint:
            for ir in range(num_r):
                fint.write(f'{r[ir]:.6f} \n')
                for iw in range(nw):
                    fint.write(f'{freq[iw]:.6f} {eps_inter[iw,ir].real:.10f} {eps_inter[iw,ir].imag:.10f} \n')


        filavgdelf = f"eps_inter_{line_cart[0]}_{line_cart[1]}_{line_cart[2]}_darkelf.dat"
        with open(filavgdelf, "w") as felf:
            felf.write('w(eV) q(eV) Re_eps Im_eps \n')
            for iw in range(nw):
                felf.write(f'{freq[iw]:.6f} 0.000000 {eps_qG_full[g_index,iw].real:.10f} {eps_qG_full[g_index,iw].imag:.10f} \n')
            for ir in range(num_r):
                for iw in range(nw):
                    felf.write(f'{freq[iw]:.6f} {r[ir]:.6f} {eps_inter[iw,ir].real:.10f} {eps_inter[iw,ir].imag:.10f} \n')


        if (scissor_correction):   
            print (f"Scissor start", flush=True)

            eps_inter_shift = np.zeros((num_r,nw+1), dtype=complex)

            fileshift = f"eps_inter_{line_cart[0]}_{line_cart[1]}_{line_cart[2]}_shifted.dat"
            with open(fileshift, "w") as fint: 
                for ir in range(num_r):
                    fint.write(f'{r[ir]:.6f} \n')
                    for iw_new in range(nw+1):
                        if iw_new < 2 :
                            val = eps_inter[0,ir]
                        else :
                            val = eps_inter[iw_new-1,ir]

                        eps_inter_shift[ir,iw_new] = val
                    
                        fint.write(f'{w_shifted[iw_new]:.6f} {val.real:.10f} {val.imag:.10f} \n')


            filedelfshift = f"eps_inter_{line_cart[0]}_{line_cart[1]}_{line_cart[2]}_darkelf_shifted.dat"
            with open(filedelfshift, "w") as felf:
                felf.write('w(eV) q(eV) Re_eps Im_eps \n')
                felf.write(f'{w_shifted[0]:.6f} 0.000000 {eps_qG_full[g_index,0].real:.10f} {eps_qG_full[g_index,0].imag:.10f} \n')
                for iw_new in range(1,nw+1):
                    felf.write(f'{w_shifted[iw_new]:.6f} 0.000000 {eps_qG_full[g_index,iw_new-1].real:.10f} {eps_qG_full[g_index,iw_new-1].imag:.10f} \n')
                for ir in range(num_r):
                    for iw_new in range(nw+1):
                        felf.write(f'{w_shifted[iw_new]:.6f} {r[ir]:.6f} {eps_inter_shift[ir,iw_new].real:.10f} {eps_inter_shift[ir,iw_new].imag:.10f} \n')


end_time = time.time()

if rank == 0 :
    now = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
    print (f"{now} Completed", flush=True)


f = paropen('times_interp.txt','w')
print(f"Initialization time: {initialization_time - start_time:.5f} seconds",file=f)
print(f"Interpolation time: {end_time - initialization_time:.5f} seconds",file=f)
print(f"Total time: {end_time - start_time:.5f} seconds",file=f)
