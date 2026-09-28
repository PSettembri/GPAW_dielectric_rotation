# GPAW_dielectric_rotation
Python scripts for the use of symmetries in the calculation of dielectric functions in GPAW

The modified symmetry.py file must be moved inside gpaw folder.
The coulomb_kernels.py and df.py inside the gpaw/response/ folder.
The gpaw folder already contains such changes,

All codes can be run in parallel using mpirun -n x gpaw python y.py, from within the conda environment where GPAW has been installed.
interp_mod.py can also be run using just mpirun -n x python interp_mod.py.

prep.py is the ground state + nscf calculation script which has to be performed before computing any dielectric function. A restarting .gpw file is created.
Important parameters are the size of the k-point grid and the wavefunctions plane wave cutoff.

eps.py is an example of the standard workflow to compute the dielectric function in GPAW. It computes the macroscopic dielectric function for q-points within the nscf calculation grid and along a line. Calculations at Gamma+G will lead to NaN. Weird behaviour at large wavevectors unless ecut is increased.

mat_rot_diff.py allows for the calculation of the microscopic dielectric function at finite momentum q, limiting the calculations to the irreducible Brillouin zone and using the crystal symmetries to rotate the dielectric function in the full reciprocal space.
Main input parameters include:
name and name_reduced - name of the restart files (.gpw excluded), if reduced is set to true, name_reduced will be used to generate the q-point grid and the associated symmetry information. Wavefunctions and eigenvalues will be gathered instead from the name.gpw calculation. The name calculation must be on a grid that is an integer multiple of the reduced one (e.g. 20x20x20 vs 10x10x10).
metal - set to True if the material has a zero - or close to zero band gap, changes the call on the Dielectric Function descriptor.
tensor - if set to true three calculations at gamma are performed to evaluate e_xx, e_yy and e_zz.
etav - dielectric function broadening parameter.
ecutv - wavefunction cutoff for the dielectric function - usually set equal to the nscf wavefunction one.
nbandsv - number of considered bands - usually set to the nscf calculation value.
method - selects the used level of approximation used, RPA or ALDA
domega0, omega2, omegamax - frequency grid descriptors, see GPAW documentation for the expression.

interp_mod.py reads the rotated dielectric function and performs an interpolation on a defined set of points.
Two modes are available:
1 - interpolation_Montecarlo_Fibonacci_parallel) generates a set of homogeneously sampled spheres using the Fibonacci sphere, then it randomly extracts a given amount of points on the spheres. Finally it interpolates the dielectric function on the generated points.
2 - interpolation_line_parallel) interpolates along a given direction in reciprocal space, usually defined in cartesian units but can also be defined in fractional.
The number of samples spheres and their radius can be changed together.
If scissor_correction is set to True additional files will be generated where the dielectric function is translated in frequency by the input parameter s (difference between experimental and theoretical band gap).
If Tensor  is set to True, for the spherical sample at Gamma the dielectric function will be substituted by the average (e_xx + e_yy + e_zz)/3, for the interpolation on a line the dielectric function at gamma is projected along the q direction (e_xx*q_x^2 + e_yy*q_y^2 + e_zz*q_z^2)
