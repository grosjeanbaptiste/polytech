"""Polytech : Bibliothèque scientifique de calcul formel et numérique.

Chaque matière constitue un contexte borné (Bounded Context) selon les
principes du Domain-Driven Design (DDD). Les contextes sont étanches et
utilisent le langage ubiquitaire propre à chaque discipline.
"""

from __future__ import annotations

from polytech import (
    algorithmique,
    chemistry,
    electromagnetisme,
    electronique,
    fluides,
    maths,
    mecanique,
    optimisation,
    raisonnement,
    signaux,
    thermique,
)

__all__ = [
    "algorithmique",
    "chemistry",
    "electromagnetisme",
    "electronique",
    "fluides",
    "maths",
    "mecanique",
    "optimisation",
    "raisonnement",
    "signaux",
    "thermique",
]
