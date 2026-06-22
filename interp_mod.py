import numpy as np
from math import sqrt
from scipy.interpolate import LinearNDInterpolator
import time
from ase.parallel import paropen, world
from mpi4py import MPI
from scipy.spatial import Delaunay

a0 = 0.5291777721092
conv_ang_ev = 1973.269 #r_max/this gives q_max in Å^-1
# Number of points on the Fibonacci
goldenRatio = (1+np.sqrt(5.))/2.

#only change the main arguments!

##############################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Initialization
def initialize():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Frequencies read from w_list file
    with open("w_list", "r") as f:
        freq = [float(line.strip()) for line in f]

    nw = len(freq)

    # Loop over the wave-vectors q

    with open("q_list", "r") as f:
        first_line = f.readline().strip()  
        nqirr, nq = map(int, first_line.split())

    eps_q = [None] * nq
    ng = np.zeros(nq,dtype=int)

    for i in range(nq):
        fileeps = f"eps_rot_{i}.csv"
        with open(fileeps, "r") as fd:
            first_line = fd.readline().strip().split()
            ng[i] = int(first_line[2])  
            eps_q[i] = np.zeros((nw, ng[i]), dtype=complex)

            for ig in range(ng[i]):
                fd.readline()
                fd.readline()  
                for iw in range(nw):
                    data_line = list(map(float, fd.readline().strip().split()))
                    eps_q[i][iw, ig] = complex(data_line[0], data_line[1])
        if rank==0:
            print(f"Reading eps file {i} of {nq}")


    with open("original_grid", "r") as fd:
        qG_eV_full = np.array([list(map(float, line.split())) for line in fd])

    npoints = len(qG_eV_full)
    eps_qG_full = np.zeros((npoints,nw), dtype=complex)
    j = 0
    for i in range(nq):
        for ig in range(ng[i]):
            eps_qG_full[j,:] = eps_q[i][:,ig]
            j += 1
    if rank==0:
        print(f"Initialization_complete!")

    initialization_time = time.time()
    
    if rank==0:
        print(f"Time to initialize: {initialization_time}.")
    return qG_eV_full, eps_qG_full, freq, nw, npoints, initialization_time

# Parallelized interpolation using Fibonacci sphere

def fibonacci(
    qG_eV_full, 
    eps_qG_full, 
    freq, 
    nw,
    npoints,
    r_min=0.001, #Radius in eV
    r_max=21500,
    num_r=700,
    logarithmic=False):

    # Parallelization in w

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    nw_local = nw // size
    start = rank * nw_local
    end = (rank + 1) * nw_local if rank != size - 1 else nw

    # Linear spacing
    if logarithmic:
        r = np.logspace(np.log10(r_min), np.log10(r_max), num_r)
    else:
        r = np.linspace(r_min,r_max,num_r)
    
    
    num_a_min = 30
    num_a_max = 30

    inter_points = []
    theta_list = []
    phi_list = []
    num_angles=[]

    if rank==0:
        print(f"Fibonacci interpolation grid")

    for ir in range(num_r):
        num_ang = int(np.round(
            num_a_min + (num_a_max - num_a_min) *
            ((r[ir] - r_min) / (r_max - r_min))**2
        ))

        idx = np.arange(num_ang)
        theta = np.arccos(1 - 2 * (idx + 0.5) / num_ang)
        phi = 2 * np.pi * idx / goldenRatio

        # Convert directly to Cartesian and store
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        theta_list.append(theta)
        phi_list.append(phi)
        num_angles.append(num_ang)
        x = r[ir] * sin_theta * cos_phi
        y = r[ir] * sin_theta * sin_phi
        z = r[ir] * cos_theta

        inter_points.append(np.column_stack((x, y, z)))
        

        if rank == 0 and ir % 50 == 0:
            print(f"[Grid] Built shell {ir+1}/{num_r}", flush=True)
            
    inter_points = np.vstack(inter_points)
    inter_points = np.ascontiguousarray(inter_points)


    if rank == 0:
        with open("interpolation_grid_Fibonacci", "w") as fd:
            for j, (x, y, z) in enumerate(inter_points):
                fd.write(f'{x:.6f} {y:.6f} {z:.6f}\n')
                if j % 500 == 0:
                    print(f'Written {j}/{len(inter_points)} points')

    # Original data points

    points = qG_eV_full[:]
    values = eps_qG_full[:,:]
    tri = Delaunay(points)

    interpolator_real = LinearNDInterpolator(tri, np.real(values[:, 0]))
    interpolator_imag = LinearNDInterpolator(tri, np.imag(values[:, 0]))

    local_eps_inter = np.empty((end-start, len(inter_points)), dtype=complex)

    if rank == 0:
        print(f'Total points: {len(inter_points)}, total frequencies: {nw}, MPI size: {size}')

    for iw, global_iw in enumerate(range(start, end)):
        
        values_iw = values[:, global_iw]
        interpolator_real.values = np.ascontiguousarray(np.real(values_iw)).reshape(-1,1)
        interpolator_imag.values = np.ascontiguousarray(np.imag(values_iw)).reshape(-1,1)
        inter_real = interpolator_real(inter_points)
        inter_real[np.isnan(inter_real)] = 0.0
        inter_imag = interpolator_imag(inter_points)
        inter_imag[np.isnan(inter_imag)] = 0.0

        local_eps_inter[iw, :] = inter_real + 1j * inter_imag
        
        if iw % 10 == 0:
            print(f'Rank {rank}: processed {iw+1}/{end-start} local frequencies')


    gathered_eps_inter = comm.gather(local_eps_inter, root=0)

    if rank == 0:
        eps_inter = np.vstack(gathered_eps_inter)
        eps_inter_avg = np.zeros((num_r,nw), dtype=complex)
        point_index = 0

        for ir in range(num_r):
            fileint = f"eps_inter_{ir}_Fibonacci.dat"
            with open(fileint, "w") as fint:
                fint.write(f'{r[ir]:.6f} \n')

                theta = theta_list[ir]
                phi = phi_list[ir]
                num_ang = num_angles[ir]

                for iangle in range(num_ang):
                    fint.write(f'{theta[iangle]:.6f} {phi[iangle] % (2*np.pi):.6f}\n')

                    j = point_index

                    for iw in range(nw):
                        fint.write(f'{eps_inter[iw,j].real:.10f} {eps_inter[iw,j].imag:.10f} \n')
                        eps_inter_avg[ir,iw] += eps_inter[iw,j]

                    point_index += 1
            if ir % 50 == 0:
                print(f'Rank 0: processed radius {ir} of {num_r}')

        for ir in range(num_r):
            eps_inter_avg[ir,:] /= num_angles[ir]

        fileavg = f"eps_inter_avg_Fibonacci.dat"
        with open(fileavg, "w") as favg:
            print(f'Rank 0: Writing {fileavg}...')
            for ir in range(num_r):
                favg.write(f'{r[ir]:.6f} \n')
                for iw in range(nw):
                    favg.write(f'{freq[iw]:.6f} {eps_inter_avg[ir,iw].real:.10f} {eps_inter_avg[ir,iw].imag:.10f} \n')
                if ir % 50 == 0:
                    print(f'Rank 0: written radius {ir} of {num_r} to {fileavg}')

        mod_k = np.linalg.norm(qG_eV_full, axis=1)
        idx = np.where(mod_k == 0.0)[0][0]

        filavgdelf = f"eps_inter_avg_Fibonacci_darkelf.dat"
        with open(filavgdelf, "w") as felf:
            print(f'Rank 0: Writing {filavgdelf}...')
            felf.write('w(eV) q(eV) Re_eps Im_eps \n')
            for iw in range(nw):
                felf.write(f'{freq[iw]:.6f} 0.000000 {eps_qG_full[idx,iw].real:.6f} {eps_qG_full[idx,iw].imag:.6f} \n')
            for ir in range(num_r):
                for iw in range(nw):
                    felf.write(f'{freq[iw]:.6f} {r[ir]:.6f} {eps_inter_avg[ir,iw].real:.10f} {eps_inter_avg[ir,iw].imag:.10f} \n')
                if ir % 50 == 0:
                    print(f'Rank 0: written radius {ir} of {num_r} to {filavgdelf}')

    return


# Interpolation on standard grid

def standard(
    qG_eV_full,
    eps_qG_full,
    freq,
    nw,
    r_min=0.01,     # Radius in eV
    r_max=10000,
    num_r=5,
    theta_min=0,        # Polar angle
    theta_max=np.pi,
    num_theta=4,
    phi_min=0,          # Azimuthal angle
    phi_max=2*np.pi,
    num_phi=5,
    logarithmic=False
    ):
    # Linear spacing
    if logarithmic:
        r = np.logspace(np.log10(r_min), np.log10(r_max), num_r)
    else:
        r = np.linspace(r_min,r_max,num_r)
        
    theta = np.linspace(theta_min,theta_max,num_theta)
    phi = np.linspace(phi_min,phi_max,num_phi)
    num_angles = num_theta*num_phi
    r_grid, theta_grid, phi_grid = np.meshgrid(r,theta,phi,indexing='ij')
    x_inter = r_grid*np.sin(theta_grid)*np.cos(phi_grid)
    y_inter = r_grid*np.sin(theta_grid)*np.sin(phi_grid)
    z_inter = r_grid*np.cos(theta_grid)

    inter_points = np.column_stack((x_inter.ravel(),y_inter.ravel(),z_inter.ravel()))

    # Parallelization in w

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    nw_local = nw // size
    start = rank * nw_local
    end = (rank + 1) * nw_local if rank != size - 1 else nw
    local_eps_inter = np.empty((end-start, len(inter_points)), dtype=complex)

    if rank==0:
        print(f"Fibonacci interpolation grid")

    if rank == 0:
        with open("interpolation_grid", "w") as fd:
            for j, (x, y, z) in enumerate(inter_points):
                fd.write(f'{x:.6f} {y:.6f} {z:.6f}\n')
                if j % 500 == 0:
                    print(f"[Grid] Written {j}/{len(inter_points)} points")

    # Original data points
    points = qG_eV_full[:]
    values = eps_qG_full[:,:]
    tri = Delaunay(points)

    interpolator_real = LinearNDInterpolator(tri, np.real(values[:, 0]))
    interpolator_imag = LinearNDInterpolator(tri, np.imag(values[:, 0]))

    for iw, global_iw in enumerate(range(start, end)):


        values_iw = values[:, global_iw]
        interpolator_real.values = np.ascontiguousarray(np.real(values_iw))
        interpolator_imag.values = np.ascontiguousarray(np.imag(values_iw))

        inter_real = interpolator_real(inter_points)
        inter_real[np.isnan(inter_real)] = 0.0
        inter_imag = interpolator_imag(inter_points)
        inter_imag[np.isnan(inter_imag)] = 0.0
        local_eps_inter[iw, :] = inter_real + 1j * inter_imag

        if iw % 10 == 0 or iw == nw-1:
            print(f'Rank {rank}: processed {iw+1}/{end-start} local frequencies')

    gathered_eps_inter = comm.gather(local_eps_inter, root=0)

    if rank==0:
        eps_inter = np.vstack(gathered_eps_inter)
        eps_inter_avg = np.zeros((num_r,nw), dtype=complex)
        norm_avg = np.zeros((num_r,nw), dtype=float)

        for ir in range(num_r):
            fileint = f"eps_inter_{ir}.dat"
            with open(fileint, "w") as fint:
                fint.write(f'{r[ir]:.6f} \n')
                for it in range(num_theta):
                    ip_range = [0] if (it == 0 or it == num_theta-1) else range(num_phi)
                    for ip in ip_range:
                        fint.write(f'{theta[it]:.6f} {phi[ip]:.6f}\n')
                        j = ir*num_angles + it*num_phi + ip
                        for iw in range(nw):
                            fint.write(f'{freq[iw]:.6f} {eps_inter[iw,j].real:.6f} {eps_inter[iw,j].imag:.6f} \n')
                            eps_inter_avg[ir,iw] += eps_inter[iw,j]*np.sin(theta[it])
                            norm_avg[ir,iw] += np.sin(theta[it])

            if ir % 1 == 0:
                print(f"[Grid] Finished writing shell {ir+1}/{num_r}", flush=True)

        # constant normalization
        # norm = 2+(num_theta-2)*num_phi
        eps_inter_avg /= norm_avg

        fileavg = f"eps_inter_avg.dat"
        with open(fileavg, "w") as favg:
            print(f'Rank 0: Writing {fileavg}...')
            for ir in range(num_r):
                favg.write(f'{r[ir]:.6f} \n')
                for iw in range(nw):
                    favg.write(f'{freq[iw]:.6f} {eps_inter_avg[ir,iw].real:.6f} {eps_inter_avg[ir,iw].imag:.6f} \n')
                if ir % 1 == 0:
                    print(f'Rank 0: written radius {ir} of {num_r} to {fileavg}')

    return

# Average from data on grid

def average(
    qG_eV_full,
    eps_qG_full,
    npoints,
    nw,
    freq,
    num_r = 10
    ):

    # Parallelization in points

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    points_per_rank = npoints // size
    start = rank * points_per_rank
    end = (rank+1)*points_per_rank if rank != size-1 else npoints


    # Compute |k| for all points
    mod_k = np.sqrt(np.sum(qG_eV_full**2, axis=1))

    r_max = np.max(mod_k)
    r = np.linspace(0,r_max,num_r)
    num_bin = num_r-1
    rcen = 0.5 * (r[1:] + r[:-1])
    
    # Local arrays
    local_count = np.zeros(num_bin, dtype=int)
    local_eps = np.zeros((nw, num_bin), dtype=complex)

    # Loop over local points
    for i in range(start, end):
        for j in range(num_bin):
            if r[j] < mod_k[i] <= r[j+1]:
                local_count[j] += 1
                local_eps[:, j] += eps_qG_full[i, :]

    # Reduce counts and sums to rank 0
    global_count = np.zeros_like(local_count)
    global_eps = np.zeros_like(local_eps)
    comm.Reduce(local_count, global_count, op=MPI.SUM, root=0)
    comm.Reduce(local_eps, global_eps, op=MPI.SUM, root=0)

    if rank == 0:
        # Compute averages
        eps_averaged = np.zeros_like(global_eps)
        for j in range(num_bin):
            if global_count[j] != 0:
                eps_averaged[:, j] = global_eps[:, j] / global_count[j]
            else:
                print(f"Warning: bin {j} empty")
                eps_averaged[:, j] = 0

        # Find idx for k=0
        idx = np.where(mod_k == 0.0)[0][0]

        fileavg = f"eps_bin_avg.dat"
        with open(fileavg, "w") as favg:
            favg.write(f'0.000000 \n')
            for iw in range(nw):
                favg.write(f'{freq[iw]:.6f} {eps_qG_full[idx,iw].real:.6f} {eps_qG_full[idx,iw].imag:.6f} \n')
            for j in range(num_bin):
                favg.write(f'{rcen[j]:.6f} \n')
                for iw in range(nw):
                    favg.write(f'{freq[iw]:.6f} {eps_averaged[iw,j].real:.6f} {eps_averaged[iw,j].imag:.6f} \n')


        fildelf = f"eps_avg_darkelf.dat"
        with open(fildelf, "w") as felf:
            felf.write('w(eV) q(eV) Re_eps Im_eps \n')
            for iw in range(nw):
                felf.write(f'{freq[iw]:.6f} 0.000000 {eps_qG_full[idx,iw].real:.6f} {eps_qG_full[idx,iw].imag:.6f} \n')
            for j in range(num_bin):
                for iw in range(nw):
                    felf.write(f'{freq[iw]:.6f} {rcen[j]:.6f} {eps_averaged[iw,j].real:.6f} {eps_averaged[iw,j].imag:.6f} \n')
        

    return

def choose(
    function,
     **kwargs
    ):
    if function == 'fibonacci':
        fibonacci( **kwargs)
    elif function == 'regular':
        standard( **kwargs)
    elif function == 'average':
        average( **kwargs)
    return

def main(
    function,
    r_min=None,
    r_max=None,
    num_r=None,
    logarithmic=None,
    theta_min=None,
    theta_max=None,
    num_theta=None,
    phi_min=None,
    phi_max=None,
    num_phi=None
):
    # Task to perform
    #Fibonacci interpolation
    #Regular interpolation
    #Average of preexisting grid
    start_time = time.time()
    qG_eV_full, eps_qG_full, freq, nw, npoints, initialization_time = initialize()

    kwargs = dict(
        qG_eV_full=qG_eV_full,
        eps_qG_full=eps_qG_full,
        freq=freq,
        nw=nw,
        npoints=npoints
    )

    if r_min is not None:
        kwargs['r_min'] = r_min
    if r_max is not None:
        kwargs['r_max'] = r_max
    if num_r is not None:
        kwargs['num_r'] = num_r
    if logarithmic is not None:
        kwargs['logarithmic'] = logarithmic
    if theta_min is not None:
        kwargs['theta_min']=theta_min
    if theta_max is not None:
        kwargs['theta_max']=theta_max
    if num_theta is not None:
        kwargs['num_theta']=num_theta
    if phi_min is not None:
        kwargs['phi_min']=phi_min
    if phi_max is not None:
        kwargs['phi_max']=phi_max
    if num_phi is not None:
        kwargs['num_phi']=num_phi

    choose(function=function, **kwargs)
    end_time = time.time()
    f = paropen('times_interp.txt','w')
    print(f"Initialization time: {initialization_time - start_time:.5f} seconds",file=f)
    print(f"Interpolation time: {end_time - initialization_time:.5f} seconds",file=f)
    print(f"Total time: {end_time - start_time:.5f} seconds",file=f)


if __name__ == "__main__":
    main(function='fibonacci', r_min = 0.001, r_max = 21000, num_r = 500, logarithmic = True)
    if rank==0:
        print("Job done!")
