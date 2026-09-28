import re

import numpy as np

from ase.utils import IOContext

from gpaw.lcaotddft.observer import TDDFTObserver


def convert_repr(r):
    # Integer
    try:
        return int(r)
    except ValueError:
        pass
    # Boolean
    b = {repr(False): False, repr(True): True}.get(r, None)
    if b is not None:
        return b
    # String
    s = r[1:-1]
    if repr(s) == r:
        return s
    raise RuntimeError('Unknown value: %s' % r)


class DipoleMomentWriter(TDDFTObserver):
    """Observer for writing time-dependent dipole moment data.

    The data is written in atomic units.

    The observer attaches to the TDDFT calculator during creation.

    Parameters
    ----------
    paw
        TDDFT calculator
    filename
        File for writing dipole moment data
    center
        If true, dipole moment is evaluated with the center of cell
        as the origin
    density
        Density type used for evaluating dipole moment.
        Use the default value for production calculations;
        others are for testing purposes.
        Possible values:
        ``'comp'``: ``rhot_g``,
        ``'pseudo'``: ``nt_sg``,
        ``'pseudocoarse'``: ``nt_sG``.
    force_new_file
        If true, new dipole moment file is created (erasing any existing one)
        even when restarting time propagation.
    interval
        Update interval. Value of 1 corresponds to evaluating and
        writing data after every propagation step.
    """
    version = 1

    def __init__(self, paw, filename: str, *,
                 center: bool = False,
                 density: str = 'comp',
                 force_new_file: bool = False,
                 interval: int = 1):
        TDDFTObserver.__init__(self, paw, interval)
        self.ioctx = IOContext()
        if paw.niter == 0 or force_new_file:
            # Initialize
            self.do_center = center
            self.density_type = density
            self.fd = self.ioctx.openfile(filename, comm=paw.world, mode='w')
            self._write_header(paw)
        else:
            # Read and continue
            self.read_header(filename)
            self.fd = self.ioctx.openfile(filename, comm=paw.world, mode='a')

    def _write(self, line):
        self.fd.write(line)
        self.fd.flush()

    def _write_header(self, paw):
        line = f'# {self.__class__.__name__}[version={self.version}]'
        line += ('(center=%s, density=%s)\n' %
                 (repr(self.do_center), repr(self.density_type)))
        line += ('# %15s %15s %22s %22s %22s\n' %
                 ('time', 'norm', 'dmx', 'dmy', 'dmz'))
        self._write(line)

    def read_header(self, filename):
        with open(filename) as f:
            line = f.readline()
        m_i = re.split("[^a-zA-Z0-9_=']+", line[2:])
        assert m_i.pop(0) == self.__class__.__name__
        for m in m_i:
            if '=' not in m:
                continue
            k, v = m.split('=')
            v = convert_repr(v)
            if k == 'version':
                assert v == self.version
                continue
            # Translate key
            k = {'center': 'do_center', 'density': 'density_type'}[k]
            setattr(self, k, v)

    def _write_init(self, paw):
        time = paw.time
        line = '# Start; Time = %.8lf\n' % time
        self._write(line)

    def _write_kick(self, paw):
        time = paw.time
        kick = paw.kick_strength
        line = '# Kick = [%22.12le, %22.12le, %22.12le]; ' % tuple(kick)
        line += 'Time = %.8lf\n' % time
        self._write(line)

    def calculate_dipole_moment(self, gd, rho_g, center=True):
        if center:
            center_v = 0.5 * gd.cell_cv.sum(0)
        else:
            center_v = np.zeros(3, dtype=float)
        r_vg = gd.get_grid_point_coordinates()
        dm_v = np.zeros(3, dtype=float)
        for v in range(3):
            dm_v[v] = - gd.integrate((r_vg[v] - center_v[v]) * rho_g)
        return dm_v

    def _write_dm(self, paw):
        time = paw.time
        density = paw.density
        if self.density_type == 'comp':
            rho_g = density.rhot_g
            gd = density.finegd
        elif self.density_type == 'pseudo':
            rho_g = density.nt_sg.sum(axis=0)
            gd = density.finegd
        elif self.density_type == 'pseudocoarse':
            rho_g = density.nt_sG.sum(axis=0)
            gd = density.gd
        else:
            raise RuntimeError('Unknown density type: %s' % self.density_type)

        norm = gd.integrate(rho_g)
        # dm = self.calculate_dipole_moment(gd, rho_g, center=self.do_center)
        dm = gd.calculate_dipole_moment(rho_g, center=self.do_center)
        if paw.hamiltonian.poisson.get_description() == 'FDTD+TDDFT':  # XXX
            dm += paw.hamiltonian.poisson.get_classical_dipole_moment()
        line = ('%20.8lf %20.8le %22.12le %22.12le %22.12le\n' %
                (time, norm, dm[0], dm[1], dm[2]))
        self._write(line)

    def _update(self, paw):
        if paw.action == 'init':
            self._write_init(paw)
        elif paw.action == 'kick':
            self._write_kick(paw)
        self._write_dm(paw)

    def __del__(self):
        self.ioctx.close()
