"""Recueil de primitives, exercice 1 : les cinquante-six primitives reunies."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from polytech.maths.application import (
    recueil_primitives_un_a,
    recueil_primitives_un_b,
    recueil_primitives_un_c,
)
from polytech.maths.application.recueil_primitives_un_c import (
    puissance_du_logarithme,
    puissance_fois_exponentielle,
    relation_45,
    relation_49,
)

if TYPE_CHECKING:
    from polytech.maths.application.resolution_exs_01 import Resolution

__all__ = ["puissance_du_logarithme", "puissance_fois_exponentielle"]
__all__ += ["relation_45", "relation_49", "resolutions"]


@cache
def resolutions() -> dict[int, Resolution]:
    """Numero de la primitive : sa resolution. Les 45 et 49 sont des recurrences."""
    return {
        **recueil_primitives_un_a.resolutions(),
        **recueil_primitives_un_b.resolutions(),
        **recueil_primitives_un_c.resolutions(),
    }
