from ase import __version__

if [int(x) for x in __version__.split('.')[:2]] > [3, 22]:
    from ase.utils.plugins import ExternalIOFormat

    gpaw_yaml = ExternalIOFormat(
        desc='GPAW-yaml output',
        code='+B',
        module='gpaw.yml',
        magic=b'#  __  _  _')
