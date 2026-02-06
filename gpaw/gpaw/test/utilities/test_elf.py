import numpy as np
import pytest
from ase.parallel import parprint

from gpaw import GPAW, restart
from gpaw.elf import ELF
from gpaw.mpi import rank


@pytest.mark.legacy
@pytest.mark.mgga
def test_utilities_elf(gpw_files):
    # Real wave functions
    atoms, calc = restart(gpw_files['h2_fd'])

    elf = ELF(calc)
    elf.update()
    elf_G = elf.get_electronic_localization_function(gridrefinement=1)
    elf_g = elf.get_electronic_localization_function(gridrefinement=2)

    # integrate the H2 bond
    if rank == 0:
        # bond area
        x0 = atoms.positions[0][0] / atoms.get_cell()[0, 0]
        x1 = atoms.positions[1][0] / atoms.get_cell()[0, 0]
        y0 = (atoms.positions[0][1] - 1.0) / atoms.get_cell()[1, 1]
        y1 = 1 - y0
        z0 = (atoms.positions[0][2] - 1.0) / atoms.get_cell()[2, 2]
        z1 = 1 - z0
        gd = calc.wfs.gd
        Gx0, Gx1 = int(gd.N_c[0] * x0), int(gd.N_c[0] * x1)
        Gy0, Gy1 = int(gd.N_c[1] * y0), int(gd.N_c[1] * y1)
        Gz0, Gz1 = int(gd.N_c[2] * z0), int(gd.N_c[2] * z1)
        finegd = calc.density.finegd
        gx0, gx1 = int(finegd.N_c[0] * x0), int(finegd.N_c[0] * x1)
        gy0, gy1 = int(finegd.N_c[1] * y0), int(finegd.N_c[1] * y1)
        gz0, gz1 = int(finegd.N_c[2] * z0), int(finegd.N_c[2] * z1)
        int1 = elf_G[Gx0:Gx1, Gy0:Gy1, Gz0:Gz1].sum() * gd.dv
        int2 = elf_g[gx0:gx1, gy0:gy1, gz0:gz1].sum() * finegd.dv
        parprint("Ints", int1, int2)
        parprint("Min, max G", np.min(elf_G), np.max(elf_G))
        parprint("Min, max g", np.min(elf_g), np.max(elf_g))
    #   The tested values (< r7887) do not seem to be correct
        assert int1 == pytest.approx(14.579199, abs=0.0001)
        assert int2 == pytest.approx(18.936101, abs=0.0001)

    # Complex wave functions
    calc = GPAW(gpw_files['bcc_li_fd'])
    elf = ELF(calc)
    elf.update()
    elf_G = elf.get_electronic_localization_function(gridrefinement=1)
