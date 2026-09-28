from pathlib import Path
from ase.build import bulk
from ase.parallel import paropen, world
from gpaw import GPAW, FermiDirac
from gpaw.response.df import DielectricFunction
import numpy as np
from ase import Atoms
from ase.optimize.bfgs import BFGS
from ase.filters import UnitCellFilter,FrechetCellFilter,StrainFilter
#from ase.constraints import ExpCellFilter
from gpaw import PW
from math import sqrt

# Ground state calculation
name = "si"
n = 20
ecut = 400

structure = bulk('Si', 'diamond', a=5.475389)

nat = len(structure)


calc = GPAW(mode=PW(ecut,dedecut='estimate'),
            kpts={'size': (20,20,20), 'gamma': True},
            occupations=FermiDirac(0.001),
            parallel={'band': 1, 'domain': 1},
            verbose=1,
            convergence={'eigenstates': 1.e-10,'energy':1.e-10},
            xc='PBE')

structure.calc = calc
structure.get_potential_energy()

# Restart Calculation with fixed density and dense kpoint sampling
calc = calc.fixed_density(
     kpts={'size': (n,n,n), 'gamma': True},
     txt='fixed.txt',verbose=1)

calc.diagonalize_full_hamiltonian(nbands=70)  # diagonalize Hamiltonian
calc.write(f'{name}_{n}_{ecut}.gpw', 'all')


#eigs = calc.get_eigenvalues(kpt=0, spin=0)
#print(eigs)
