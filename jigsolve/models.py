from dataclasses import dataclass, field
import numpy as np
from typing import Tuple

@dataclass(eq=False, order=False, frozen=True)
class PuzzlePiece:
    img: np.ndarray = field(repr=False)
    mask: np.ndarray = field(repr=False)
    hist: Tuple[np.ndarray] = field(repr=False)
    origin: Tuple[int]
    img_rot: float
    box: Tuple[int]
    edges: Tuple[int]
