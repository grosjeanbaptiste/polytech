"""Les lois de Newton appliquees : machine d'Atwood, plan incline, cercle.

Source : syllabus « Notions fondamentales » de « Physique generale »,
paragraphes 4.6 et 4.7, equations (4.16), (4.17), (4.21), (4.22), (4.25) et
(4.26), pages 43 et 44.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def atwood(m1: Any, m2: Any, g: Any) -> tuple[Any, Any]:  # noqa: ANN401 -- nombres ou expressions sympy
    """(4.16) a = (m2 - m1) g / (m1 + m2) et (4.17) T = 2 m1 m2 g / (m1 + m2)."""
    return (m2 - m1) * g / (m1 + m2), 2 * m1 * m2 * g / (m1 + m2)


@dataclass(frozen=True)
class PlanIncline:
    """Un bloc sur un plan incline sans frottement."""

    tension: Any
    """(4.21) la tension de la corde qui le retient."""
    normale: Any
    """(4.22) la reaction du plan, qui n'est pas le poids."""
    acceleration: Any
    """(4.25) l'acceleration le long du plan, la corde coupee."""


def plan_incline(m: Any, g: Any, theta: Any) -> PlanIncline:  # noqa: ANN401 -- nombres ou expressions sympy
    """Les forces et l'acceleration d'un bloc de masse m sur un plan d'angle theta."""
    if isinstance(theta, (int, float)):
        import math  # noqa: PLC0415 -- le chemin numerique

        sin, cos = math.sin(theta), math.cos(theta)
    else:
        import sympy as sp  # noqa: PLC0415 -- le chemin symbolique

        sin, cos = sp.sin(theta), sp.cos(theta)
    return PlanIncline(m * g * sin, m * g * cos, -g * sin)


def acceleration_centripete(v: Any, r: Any) -> Any:  # noqa: ANN401 -- nombre ou expression sympy
    """(4.26) a = v^2 / r, dirigee vers le centre ; leve pour un rayon nul."""
    if isinstance(r, (int, float)) and r == 0:
        msg = "un mouvement circulaire demande un rayon non nul"
        raise ValueError(msg)
    return v**2 / r
