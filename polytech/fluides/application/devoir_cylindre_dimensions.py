"""Devoir cote de fluides : l'analyse dimensionnelle par les variables repetees.

La methode est celle du syllabus du cours
(`MecaniqueFluides/SYLLABUS/Syllabus_Theoreme_Vaschy-Buckingham.pdf`,
extrait du chapitre 7 de Cengel), deja portee par `analyse_dimensionnelle`
et `variables_repetees`. La pression en un point de la paroi depend de
`rho, U, D, mu` : cinq parametres, trois dimensions (M, L, T), donc deux
groupes. `rho, U, D` sont repetes : ils portent les trois dimensions sans
former de groupe a eux seuls.

Dans un gaz rapide, la celerite du son `c` entre a son tour, et avec elle
un troisieme groupe, le Mach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.fluides.domain.analyse_dimensionnelle import nombre_de_groupes
from polytech.fluides.domain.variables_repetees import (
    Dimensions,
    Groupe,
    dimensions_du_produit,
    groupes_pi,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import sympy as sp

PARAMETRES: Final[dict[str, Dimensions]] = {
    "dp": {"M": 1, "L": -1, "T": -2},
    "rho": {"M": 1, "L": -3},
    "U": {"L": 1, "T": -1},
    "D": {"L": 1},
    "mu": {"M": 1, "L": -1, "T": -1},
}
"""`p - p_inf` d'abord : c'est la variable dependante."""

CELERITE: Final[dict[str, Dimensions]] = {"c": {"L": 1, "T": -1}}
REPETES: Final = ("rho", "U", "D")
DIMENSIONS_PRIMAIRES: Final = 3


def _parametres(*, compressible: bool) -> dict[str, Dimensions]:
    return {**PARAMETRES, **CELERITE} if compressible else dict(PARAMETRES)


def nombre_de_groupes_attendu(*, compressible: bool = False) -> int:
    """`k = n - j` : Vaschy-Buckingham."""
    return nombre_de_groupes(
        len(_parametres(compressible=compressible)), DIMENSIONS_PRIMAIRES
    )


def groupes(*, compressible: bool = False) -> tuple[Groupe, ...]:
    """Les groupes, celui de la pression en tete."""
    return groupes_pi("dp", REPETES, _parametres(compressible=compressible))


def dimensions(groupe: Mapping[str, sp.Expr | int]) -> dict[str, sp.Expr]:
    """Les dimensions d'un groupe ; vide s'il est sans dimension."""
    return dimensions_du_produit(groupe, _parametres(compressible=True))
