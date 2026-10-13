"""Les symboles du recueil de primitives du syllabus de 2011.

Les parametres sont pris strictement positifs et `n` entier positif :
c'est le cas general de chaque enonce, et les cas particuliers que le
syllabus signale (`b = 0`, `p = -1`) sont dits dans le corrige.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.application.resolution_exs_01 import A, B
from polytech.maths.domain.symboles import ENTIER as N
from polytech.maths.domain.symboles import VARIABLE as X

__all__ = ["A", "B", "H", "K", "N", "P", "Q", "R", "W", "X"]

R = sp.Rational
P: Final = sp.Symbol("p", positive=True)
Q: Final = sp.Symbol("q", positive=True)
"""La seconde borne de l'exercice 10 e) : `p < x < q`."""
K: Final = sp.Symbol("k", positive=True)
"""`k = 4 ln 100`, l'exposant de l'exercice 5 b) : `100^(-4x) = e^(-kx)`."""
W: Final = sp.Symbol("w", positive=True)
H: Final = sp.Symbol("h", positive=True)
"""La pulsation et le dephasage de l'exercice 8 j), dont la variable est le temps."""
