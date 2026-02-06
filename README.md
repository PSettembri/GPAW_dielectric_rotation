# GPAW_dielectric_rotation
Python scripts for the use of symmetries in the calculation of dielectric functions in GPAW

The modified symmetry.py file must be moved inside gpaw folder.
The coulomb_kernels.py and df.py inside gpaw/response/ .

eps.py is an example of the standard workflow to compute the dielectric function in GPAW.
mat_rot_diff_XX_YY.py allows for the calculation of the microscopic dielectric function at finite momentum q, limiting the calculations to the irreducible Brillouin zone and using the crystal symmetries to rotate the dielectric function in the full reciprocal space.
interp_mod_XX_YY.py reads the rotated dielectric function, generates a set of homogeneously sampled spheres and interpolates the dielectric function on the generated points.