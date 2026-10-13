"""Recueil de primitives, exercices 2 a 11 : les soixante-six resolutions reunies."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from polytech.maths.application import (
    recueil_primitives_suite_a,
    recueil_primitives_suite_b,
    recueil_primitives_suite_c,
    recueil_primitives_suite_d,
)

if TYPE_CHECKING:
    from polytech.maths.application.resolution_exs_01 import Resolution


@cache
def resolutions() -> dict[str, Resolution]:
    """Cle de la primitive -- `2a`, `10aPlus` -- : sa resolution."""
    return {
        **recueil_primitives_suite_a.rationnelles(),
        **recueil_primitives_suite_a.exponentielles(),
        **recueil_primitives_suite_a.diverses(),
        **recueil_primitives_suite_b.exercice_6(),
        **recueil_primitives_suite_b.exercice_7(),
        **recueil_primitives_suite_c.par_un_changement(),
        **recueil_primitives_suite_c.par_une_reecriture(),
        **recueil_primitives_suite_d.exercice_9(),
        **recueil_primitives_suite_d.exercice_10(),
        **recueil_primitives_suite_d.exercice_11(),
    }
