"""Modelisation, probleme VIIb : quatre conditions de plus sur les dates.

Chaque condition de l'enonce est ecrite deux fois : comme une phrase
sur les dates de debut, et comme les contraintes lineaires de sa
planche. Les 720 plannings disent si les deux ecritures s'accordent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.optimisation.application.modele_maintenance import (
    CHARGES,
    Debuts,
    classement,
)

if TYPE_CHECKING:
    from collections.abc import Callable

DUREE: Final = len(CHARGES[1])
CONDITIONS: Final[dict[str, Callable[[Debuts], bool]]] = {
    "A3 pas avant A1": lambda d: d[2] >= d[0],
    "A4 en meme temps que A3": lambda d: d[3] == d[2],
    "A4 et A5 pas la meme semaine": lambda d: d[3] != d[4],
    "A5 apres la fin de A1": lambda d: d[4] >= d[0] + DUREE,
}
"""Planche 38, sur les dates de debut `(A1, ..., A5)`."""
Terme = tuple[int, int]
Contrainte = tuple[tuple[Terme, ...], str, tuple[Terme, ...]]
"""`somme des delta de gauche  (<=, =, >=)  somme des delta de droite`."""
UN: Final[tuple[Terme, ...]] = ((0, 0),)
"""Un second membre egal a 1 : le terme `(0, 0)` vaut toujours 1."""
PLANCHES: Final[dict[str, tuple[Contrainte, ...]]] = {
    "A3 pas avant A1": (
        (((3, 2),), "<=", ((1, 1), (1, 2))),
        (((3, 3),), "<=", ((1, 1), (1, 2), (1, 3))),
    ),
    "A4 en meme temps que A3": (
        (((4, 6),), "=", ()),
        *((((3, k),), "=", ((4, k),)) for k in (2, 3, 4, 5)),
    ),
    "A4 et A5 pas la meme semaine": tuple(
        (((4, k), (5, k)), "<=", UN) for k in (3, 4, 5)
    ),
    "A5 apres la fin de A1": (
        (((1, 3),), "=", ()),
        (((1, 4),), "=", ()),
        (((5, 3),), "=", ()),
        (((5, 4), (5, 5)), ">=", ((1, 1),)),
        (((5, 5),), ">=", ((1, 2),)),
    ),
}
"""Planches 40 a 43, recopiees."""


def _somme(termes: tuple[Terme, ...], debuts: Debuts) -> int:
    """La valeur d'une somme de `delta_ij` pour ce planning."""
    return sum(
        1 if terme == (0, 0) else int(debuts[terme[0] - 1] == terme[1])
        for terme in termes
    )


def respecte_la_planche(condition: str, debuts: Debuts) -> bool:
    """Vrai si le planning satisfait toutes les contraintes de la planche."""
    compare = {"<=": int.__le__, "=": int.__eq__, ">=": int.__ge__}
    return all(
        compare[sens](_somme(gauche, debuts), _somme(droite, debuts))
        for gauche, sens, droite in PLANCHES[condition]
    )


def bilan(condition: str) -> tuple[int, int, int]:
    """Sous la condition : plannings permis, cout minimal, plus petite pointe."""
    if condition not in CONDITIONS:
        msg = f"condition inconnue : {condition}"
        raise KeyError(msg)
    permis = [ligne for ligne in classement() if CONDITIONS[condition](ligne[2])]
    return len(permis), permis[0][0], min(pointe for _, pointe, _ in permis)
