from typing import Tuple

from .wrapper import HybridXC

__all__ = ['HybridXC']


def parse_name(name: str) -> Tuple[str, float, float]:
    if name == 'EXX':
        return 'null', 1.0, 0.0
    if name == 'PBE0':
        return 'HYB_GGA_XC_PBEH', 0.25, 0.0
    if name == 'HSE03':
        return 'HYB_GGA_XC_HSE03', 0.25, 0.106
    if name == 'HSE06':
        return 'HYB_GGA_XC_HSE06', 0.25, 0.11
    if name == 'B3LYP':
        return 'HYB_GGA_XC_B3LYP', 0.2, 0.0
    assert False
