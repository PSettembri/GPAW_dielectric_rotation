GPAW_dielectric_rotation
Modified version from Federico Giannessi. Main difference: parallelization over q points, a few quality of life updates.

Python scripts for the use of symmetries in the calculation of dielectric functions in GPAW

The modified symmetry.py file must be moved inside gpaw folder. The coulomb_kernels.py and df.py inside the gpaw/response/ folder. The gpaw folder already contains such changes,

All codes can be run in parallel using mpirun -n x gpaw python y.py, from within the conda environment where GPAW has been installed. interp_mod.py can also be run using just mpirun -n x python interp_mod.py.


The `prep.py` script prepares the auxiliary files required by the modified GPAW response workflow. It must be run **serially on a single processor**.

Starting from a GPAW `.gpw` ground-state file, the script:

* reads the crystal structure and reciprocal lattice vectors;
* extracts the irreducible and full Brillouin-zone k-point grids and their weights;
* determines the mapping between full-zone and irreducible k-points using the crystal symmetries;
* reads the symmetry matrices from `symmetries.txt`;
* accounts for time-reversal symmetry when constructing the symmetry-operation mappings;
* determines the integer reciprocal-lattice translations required to map full-zone k-points onto their symmetry-equivalent irreducible k-points;
* constructs the corresponding inverse symmetry mappings;
* generates the nonlinear frequency grid used by the response calculation;
* writes the resulting mappings and auxiliary data to files for use by the subsequent modified-GPAW calculations.

The main output files are:

```text
U_scc.npy          Symmetry-operation matrices
U_scc_inv.npy      Inverse symmetry-operation matrices
sym_k_2.npy        Symmetry operation for each full-zone k-point
sym_k_2_inv.npy    Inverse symmetry operation for each k-point
N_vec_first.npy    Reciprocal-lattice translation for the forward mapping
N_vec.npy          Reciprocal-lattice translation for the inverse mapping
b_vectors.dat      Reciprocal lattice vectors
w_list             Frequency grid
qirr_rot.txt       Human-readable k-point/symmetry mapping
```

These files provide the information required to reconstruct quantities calculated on the irreducible Brillouin zone over the full Brillouin zone, including the corresponding symmetry transformations and reciprocal-lattice translations.

### Input parameters

The main parameters are defined near the beginning of the script:

```python
name = "al"
domega0 = 0.1
omega2 = 100
omegamax = 1050
```

where `name` specifies the input `.gpw` file and `domega0`, `omega2`, and `omegamax` define the nonlinear frequency grid.

The script expects the corresponding GPAW file

```text
<name>.gpw
```

and the symmetry operations in

```text
symmetries.txt
```

### Usage

Run the script once using a single processor:

```bash
gpaw python prep.py
```

The generated files are then used by the subsequent response calculations.


