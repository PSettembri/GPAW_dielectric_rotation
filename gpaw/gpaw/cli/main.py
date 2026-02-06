"""GPAW command-line tool."""
import os
import subprocess
import sys


commands = [
    ('run', 'gpaw.cli.run'),
    ('info', 'gpaw.cli.info'),
    ('test', 'gpaw.cli.test'),
    ('dos', 'gpaw.cli.dos'),
    ('gpw', 'gpaw.cli.gpw'),
    ('completion', 'gpaw.cli.completion'),
    ('atom', 'gpaw.atom.aeatom'),
    ('diag', 'gpaw.fulldiag'),
    # ('quick', 'gpaw.cli.quick'),
    ('python', 'gpaw.cli.python'),
    ('sbatch', 'gpaw.cli.sbatch'),
    ('dataset', 'gpaw.atom.generator2'),
    ('symmetry', 'gpaw.symmetry'),
    ('install-data', 'gpaw.cli.install_data')]


def hook(parser, args):
    parser.add_argument('-P', '--parallel', type=int, metavar='N',
                        help='Run on N CPUs.')
    args = parser.parse_args(args)

    if args.command == 'python':
        args.traceback = True

    if hasattr(args, 'dry_run'):
        N = int(args.dry_run)
        if N:
            import gpaw
            gpaw.dry_run = N
            import gpaw.mpi as mpi
            mpi.world = mpi.SerialCommunicator()
            mpi.world.size = N

    if args.parallel:
        from gpaw.mpi import have_mpi, world
        if have_mpi and world.size == 1 and args.parallel > 1:
            py = sys.executable
        elif not have_mpi:
            py = 'gpaw-python'
        else:
            py = ''

        if py:
            # Start again in parallel:
            arguments = ['mpiexec', '-np', str(args.parallel), py]
            if args.command == 'python' and args.debug:
                arguments.append('-d')
            arguments += ['-m', 'gpaw']
            arguments += sys.argv[1:]

            extra = os.environ.get('GPAW_MPI_OPTIONS')
            if extra:
                arguments[1:1] = extra.split()

            # Use a clean set of environment variables without any MPI stuff:
            p = subprocess.run(arguments, check=not True, env=os.environ)
            sys.exit(p.returncode)

    return args


def main(args=None):
    from gpaw import all_lazy_imports, broadcast_imports, __getattr__
    with broadcast_imports:
        for attr in all_lazy_imports:
            __getattr__(attr)

        from ase.cli.main import main as ase_main
        from gpaw import __version__

    ase_main('gpaw', 'GPAW command-line tool', __version__,
             commands, hook, args)
