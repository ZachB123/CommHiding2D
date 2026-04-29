from dataclasses import dataclass
from typing import Callable, Optional

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

@dataclass(frozen=True)
class Gemm2DInnerLoopConfiguration:
    make_buffer: Optional[Callable]
    persistent_buffer: bool
    loopback: bool
    tiles: Callable
    set_c_tile: Optional[Callable]
    make_inner_c_matrix: Optional[Callable]
    reduce_scatter_finalize: Optional[Callable]

@dataclass(frozen=True)
class Gemm2DAlgorithmConfiguration:
    group_param: Callable
    divisibility: DivisibiltyRequirements
    distribution: DistributionFunctions
    get_local_indices: Callable
    flatten_gather: bool = False