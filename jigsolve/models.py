from dataclasses import dataclass, field
import numpy as np

@dataclass(eq=False, order=False, frozen=True)
class PuzzlePiece:
    img: np.ndarray = field(repr=False)
    mask: np.ndarray = field(repr=False)
    img_rot: float
    box: tuple[int]
    edges: tuple[int]
    colors: tuple[np.ndarray] = field(repr=False)
