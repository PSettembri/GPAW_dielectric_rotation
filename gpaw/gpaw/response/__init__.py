"""GPAW Response core functionality."""
from __future__ import annotations
from typing import Union
from gpaw.calculator import GPAW as OldGPAW
from gpaw.new.ase_interface import ASECalculator as NewGPAW
from .groundstate import ResponseGroundStateAdapter, GPWFilename  # noqa
from .context import ResponseContext, TXTFilename, timer  # noqa

__all__ = ['ResponseGroundStateAdapter', 'GPWFilename',
           'ResponseContext', 'TXTFilename', 'timer']


GPAWCalculator = Union[OldGPAW, NewGPAW]
ResponseGroundStateAdaptable = Union[ResponseGroundStateAdapter,
                                     GPAWCalculator,
                                     GPWFilename]


def ensure_gs_and_context(gs: ResponseGroundStateAdaptable,
                          context: ResponseContext | TXTFilename = '-')\
        -> tuple[ResponseGroundStateAdapter, ResponseContext]:
    return ensure_gs(gs), ensure_context(context)


def ensure_gs(gs: ResponseGroundStateAdaptable) -> ResponseGroundStateAdapter:
    if not isinstance(gs, ResponseGroundStateAdapter):
        if isinstance(gs, (OldGPAW, NewGPAW)):
            gs = ResponseGroundStateAdapter(calc=gs)
        else:  # gs is a GPWFilename
            gs = ResponseGroundStateAdapter.from_gpw_file(gpw=gs)
    return gs


def ensure_context(context: ResponseContext | TXTFilename) -> ResponseContext:
    if not isinstance(context, ResponseContext):
        context = ResponseContext(txt=context)
    return context
