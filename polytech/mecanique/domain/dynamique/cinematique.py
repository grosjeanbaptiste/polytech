"""La cinematique a acceleration constante (MRUA).

Source : syllabus « Notions fondamentales » de « Physique generale »,
paragraphe 2.3, equations (2.10), (2.14) et (2.15), pages 17 et 18. Les
fonctions acceptent des nombres ou des expressions sympy : une meme formule
sert le calcul chiffre et la verification symbolique.
"""

from __future__ import annotations

import math
from typing import Any


def vitesse_mrua(v0: Any, a: Any, t: Any) -> Any:  # noqa: ANN401 -- nombre ou expression sympy
    """(2.10) v = v0 + a t."""
    return v0 + a * t


def position_mrua(x0: Any, v0: Any, a: Any, t: Any) -> Any:  # noqa: ANN401 -- nombre ou expression sympy
    """(2.14) x = x0 + v0 t + a t^2 / 2."""
    return x0 + v0 * t + a * t**2 / 2


def vitesse_par_la_position(v0: Any, a: Any, x0: Any, x: Any) -> Any:  # noqa: ANN401 -- nombre ou expression sympy
    """(2.15) |v| = racine(v0^2 + 2 a (x - x0)) ; leve si x n'est jamais atteinte."""
    carre = v0**2 + 2 * a * (x - x0)
    if isinstance(carre, (int, float)):
        if carre < 0:
            msg = f"position {x} jamais atteinte : v^2 = {carre} < 0"
            raise ValueError(msg)
        return math.sqrt(carre)
    import sympy as sp  # noqa: PLC0415 -- le chemin symbolique seulement

    return sp.sqrt(carre)
