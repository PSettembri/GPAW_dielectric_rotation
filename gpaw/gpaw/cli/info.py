import os
import sys

from ase.utils import import_module, search_current_git_hash

import gpaw.cgpaw as cgpaw
import gpaw
import gpaw.fftw as fftw
from gpaw.mpi import have_mpi, rank
from gpaw.new.c import GPU_AWARE_MPI, GPU_ENABLED
from gpaw.utilities import compiled_with_libvdwxc, compiled_with_sl
from gpaw.utilities.elpa import LibElpa
from gpaw.gpu import cupy


def info():
    """Show versions of GPAW and its dependencies."""
    results = [('python-' + sys.version.split()[0], sys.executable)]
    for name in ['gpaw', 'ase', 'numpy', 'scipy']:
        try:
            module = import_module(name)
        except ImportError:
            results.append((name, False))
        else:
            # Search for git hash
            githash = search_current_git_hash(module)
            if githash is None:
                githash = ''
            else:
                githash = f'-{githash:.10}'
            results.append((name + '-' + module.__version__ + githash,
                            module.__file__.rsplit('/', 1)[0] + '/'))

    libs = gpaw.get_libraries()

    libxc = libs['libxc']
    if libxc:
        results.append((f'libxc-{libxc}', True))
    else:
        results.append(('libxc', False))

    if hasattr(cgpaw, 'githash'):
        githash = f'-{cgpaw.githash():.10}'
    else:
        githash = ''

    results.append(('_gpaw' + githash,
                    os.path.normpath(getattr(cgpaw._gpaw, '__file__',
                                             'built-in'))))

    results.append(('MPI enabled', have_mpi))
    results.append(('OpenMP enabled', cgpaw.have_openmp))
    results.append(('GPU enabled', GPU_ENABLED))
    results.append(('GPU-aware MPI', GPU_AWARE_MPI))
    results.append(('CUPY', cupy.__file__))
    if have_mpi:
        have_sl = compiled_with_sl()
        have_elpa = LibElpa.have_elpa()
        if have_elpa:
            version = LibElpa.api_version()
            if version is None:
                version = 'unknown, at most 2018.xx'
            have_elpa = f'yes; version: {version}'
    else:
        have_sl = have_elpa = 'no (MPI unavailable)'

    if not hasattr(cgpaw, 'mmm'):
        results.append(('BLAS', 'using scipy.linalg.blas and numpy.dot()'))

    results.append(('scalapack', have_sl))
    results.append(('Elpa', have_elpa))

    have_fftw = fftw.have_fftw()
    results.append(('FFTW', have_fftw))
    results.append(('libvdwxc', compiled_with_libvdwxc()))
    for i, path in enumerate(gpaw.setup_paths):
        results.append((f'PAW-datasets ({i + 1})', str(path)))

    if rank != 0:
        return

    lines = [(a, b if isinstance(b, str) else ['no', 'yes'][b])
             for a, b in results]
    n1 = max(len(a) for a, _ in lines)
    n2 = max(len(b) for _, b in lines)
    print(' ' + '-' * (n1 + 4 + n2))
    for a, b in lines:
        print(f'| {a:{n1}}  {b:{n2}} |')
    print(' ' + '-' * (n1 + 4 + n2))


class CLICommand:
    """Show versions of GPAW and its dependencies"""

    @staticmethod
    def add_arguments(parser):
        pass

    @staticmethod
    def run(args):
        info()
