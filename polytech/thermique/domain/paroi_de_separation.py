"""Le coefficient global de transmission entre deux fluides qu'une paroi separe.

Syllabus de thermique, tome 2, deuxieme partie, sections 1.1 et 1.4.
`phi = K_gl (t_1 - t_2)` : le flux rencontre trois resistances en serie,
`R_gl = R_1 + R_p + R_2`.

Pour un tube, la densite de flux ne se conserve pas d'une face a
l'autre ; le flux lineique, si. Le coefficient se rapporte donc a une
surface qu'il faut nommer -- l'exterieure (1.4) ou l'interieure (1.5) --
et `K_gl,e S_e = K_gl,i S_i` (1.3). La relation (1.6) vaut pour une
paroi quelconque et contient les precedentes.

La difference de temperature se partage en chutes partielles « qui sont
entre elles comme le rapport des resistances » (section 1.4) : presque
tout l'ecart se trouve aux bornes de la plus grande.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class Reference(Enum):
    """La surface a laquelle le coefficient d'un tube se rapporte."""

    INTERIEURE = "K_gl,i, relation (1.5)"
    EXTERIEURE = "K_gl,e, relation (1.4)"


def coefficient_global_rapporte(
    reference: float,
    *,
    cotes: Sequence[tuple[float, float]],
    resistance_de_paroi: float,
) -> float:
    """(1.6) : `1/K_gl = somme(S_ref / (K S)) + R_p S_ref`, `cotes` en `(K, S)`."""
    if reference <= 0:
        msg = f"surface de reference nulle ou negative : {reference} m^2"
        raise ValueError(msg)
    fluides = sum(1 / (echange * surface) for echange, surface in cotes)
    return 1 / (reference * (fluides + resistance_de_paroi))


def coefficient_global_de_tube(
    interieur: float,
    exterieur: float,
    *,
    rayons: tuple[float, float],
    conductivite: float,
    reference: Reference,
) -> float:
    """(1.4) ou (1.5), selon la surface de `reference` ; `rayons = (r_i, r_e)`."""
    interne, externe = rayons
    if externe <= interne:
        msg = f"rayon exterieur non superieur a l'interieur : {externe} <= {interne}"
        raise ValueError(msg)
    surfaces = (2 * math.pi * interne, 2 * math.pi * externe)
    choisie = surfaces[0] if reference is Reference.INTERIEURE else surfaces[1]
    paroi = math.log(externe / interne) / (2 * math.pi * conductivite)
    return coefficient_global_rapporte(
        choisie,
        cotes=((interieur, surfaces[0]), (exterieur, surfaces[1])),
        resistance_de_paroi=paroi,
    )


def chutes_partielles(ecart: float, resistances: Sequence[float]) -> tuple[float, ...]:
    """Les chutes de temperature en serie, proportionnelles aux resistances."""
    for resistance in resistances:
        if resistance < 0:
            msg = f"resistance negative : {resistance}"
            raise ValueError(msg)
    totale = sum(resistances)
    return tuple(ecart * resistance / totale for resistance in resistances)
