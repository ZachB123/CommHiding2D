from dataclasses import dataclass
from typing import Callable

from enums import GemmDimension


@dataclass(frozen=True)
class DivisibiltyRequirements():
    m_divisibility: GemmDimension
    k_divisibility: GemmDimension
    n_divisibility: GemmDimension

    def __repr__(self) -> str:
        return (
            f"({self.m_divisibility.name}, "
            f"{self.k_divisibility.name}, "
            f"{self.n_divisibility.name})"
        )

@dataclass(frozen=True)
class DistributionFunctions:
    A_distribution: Callable
    B_distribution: Callable
    C_distribution: Callable

    def __repr__(self) -> str:
        return (
            "DistributionFunctions("
            f"A={self.A_distribution.__name__}, "
            f"B={self.B_distribution.__name__}, "
            f"C={self.C_distribution.__name__}"
            ")"
        )
    
@dataclass(frozen=True)
class CurrentTiles:
    A_curr: Callable
    B_curr: Callable
    C_curr: Callable