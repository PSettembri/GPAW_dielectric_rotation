from pathlib import Path
from ase.build import bulk
from ase.parallel import paropen, world
from gpaw import GPAW, FermiDirac, restart
from gpaw.response.df import DielectricFunction, CustomizableDielectricFunction
import numpy as np
from ase import Atoms
from ase.optimize.bfgs import BFGS
from ase.filters import UnitCellFilter,FrechetCellFilter,StrainFilter
#from ase.constraints import ExpCellFilter
from gpaw import PW
from math import sqrt

# Getting absorption spectrum
f = paropen('q_list','w')
si,calc = restart('si.gpw')
for i in range(1):
    df = DielectricFunction(calc='si.gpw',
            eta=0.2,
            frequencies={'type': 'nonlinear', 'domega0': 0.01, 'omegamax': 10},
            ecut=100,
            nbands=70,
            txt='out_df_%d.txt' %i)

    q_c = [0, 2*i/30, 2*i/30]

    df.get_dielectric_function(q_c=q_c, filename='si_eps_%d.csv' % i )     
    cell_cv = si.get_cell()
    bcell_cv = 2 * np.pi * np.linalg.inv(cell_cv).T
    q_v = np.dot(q_c, bcell_cv)
    print(sqrt(np.inner(q_v, q_v)), file=f)



f.close()



