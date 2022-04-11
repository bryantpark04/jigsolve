from dataclasses import dataclass, field
import numpy as np
from typing import Tuple

@dataclass(eq=False, order=False, frozen=True)
class PuzzlePiece:
    combined: np.ndarray = field(repr=False)
    hist: Tuple[np.ndarray] = field(repr=False)
    origin: Tuple[int]
    rot: float
    box: Tuple[int]
    edges: Tuple[int]
