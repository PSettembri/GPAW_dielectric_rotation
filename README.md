GPAW_dielectric_rotation
Modified version from Federico Giannessi. Main difference: parallelization over q points, a few quality of life updates.

Python scripts for the use of symmetries in the calculation of dielectric functions in GPAW

The modified symmetry.py file must be moved inside gpaw folder. The coulomb_kernels.py and df.py inside the gpaw/response/ folder. The gpaw folder already contains such changes,

All codes can be run in parallel using mpirun -n x gpaw python y.py, from within the conda environment where GPAW has been installed. interp_mod.py can also be run using just mpirun -n x python interp_mod.py.

prep.py prepares the auxiliary files required by the modified GPAW response calculation. It reads the .gpw ground-state file and crystal symmetries, determines the irreducible/full Brillouin-zone k-point mappings, and generates the frequency grid.
The generated files are then used by the subsequent modified-GPAW calculations.

mat_para.py calculates the dielectric function on the irreducible Brillouin zone using GPAW and reconstructs the corresponding dielectric data on the full k-point grid using crystal symmetries.
The calculation is distributed over MPI processes, with all processes working on the same irreducible q-point at once. The main parameters are set in the calculate_df() call at the end of the script.
The script also generates the frequency grid, q-point list, rotated G-vectors, and the full reconstructed dielectric-function grid used by the subsequent interpolation/post-processing steps.

mat_diff.py performs the same dielectric-function calculation as mat.py, but distributes the irreducible q-points across MPI processes, with each process handling one q-point at a time.

interp_new.py reads the rotated dielectric-function data and interpolates it onto user-defined q-space grids. Four interpolation/averaging modes are available:
fibonacci — interpolates on spherical shells sampled using a Fibonacci sphere and computes the angular average.
regular — interpolates on a regular spherical grid defined by radial, polar, and azimuthal points.
average — directly averages the existing dielectric-function data in radial q-bins.
direction — interpolates along one or more specified crystallographic directions.
The interpolation can be performed with linear or logarithmic radial spacing. The calculations are MPI-parallelized over frequencies.
The main parameters are set in the main() call at the end of the script:

main(
    function='fibonacci',
    r_min=264,
    r_max=14500,
    num_r=381,
    logarithmic=False)
For directional interpolation, the name of the .gpw file must be give and crystallographic directions can be specified through crystal_dirs, e.g.
crystal_dirs=((1, 0, 0), (0, 0, 1), (1, 1, 1))
