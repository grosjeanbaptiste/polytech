"""Modelisations particulieres : un plan de production, juge sur l'enonce.

Cinq mois, une demande connue, un cout unitaire par mois, un cout de
stockage. Onze variantes ajoutent chacune une regle. Ce module ne passe
par aucun programme lineaire : il dresse tous les plans entiers et
applique a chacun la regle telle que l'enonce la dit.
"""

from __future__ import annotations

import itertools
from functools import cache
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable

Plan = tuple[int, ...]
DEMANDES: Final = (5, 6, 8, 6, 5)
ALTERNATIVE: Final = (7, 5, 3, 9, 6)
COUTS: Final = (1, 4, 2, 4, 2)
"""Planches 47 et 68."""
ECART_PERMIS, ECART_EXIGE, MOIS_DE_FABRICATION = 2, 3, 3
"""Planches 55, 57 et 59 : les seuils des variantes V, VI et VII."""


def _mois_ouverts(plan: Plan) -> int:
    return sum(1 for quantite in plan if quantite)


def _variations(plan: Plan, hausse: int, baisse: int) -> int:
    """Ce que coutent les changements de cadence d'un mois au suivant."""
    return sum(
        hausse * (b - a) if b > a else baisse * (a - b)
        for a, b in itertools.pairwise(plan)
    )


SUPPLEMENTS: Final[dict[str, Callable[[Plan], int]]] = {
    "II": lambda p: _variations(p, 2, 0),
    "III": lambda p: _variations(p, 4, 2),
    "IV": lambda p: 2 * (max(p) - min(p)),
}
"""Ce qu'une variante ajoute au cout de production et de stockage."""
CONDITIONS: Final[dict[str, Callable[[Plan], bool]]] = {
    "II": lambda p: all(b >= a for a, b in itertools.pairwise(p)),
    "V": lambda p: abs(p[0] - p[1]) <= ECART_PERMIS and abs(p[2] - p[3]) >= ECART_EXIGE,
    "VI": lambda p: _mois_ouverts(p) <= MOIS_DE_FABRICATION,
    "VII": lambda p: _mois_ouverts(p) == MOIS_DE_FABRICATION,
    "VIII": lambda p: p[3] > 0 or not (p[0] and p[1] and p[2]),
    "IX": lambda p: p[3] == 0 or bool(p[0] and p[1] and p[2]),
    "X": lambda p: p[3] == 0 or bool(p[1] or p[2]),
}
"""Ce qu'une variante exige d'un plan."""
VARIANTES: Final = (
    "I",
    "stock",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
)


def avances_cumulees(demandes: Plan) -> int:
    """La somme des demandes cumulees des quatre premiers mois."""
    return sum(itertools.accumulate(demandes[:-1]))


@cache
def plans(demandes: Plan = DEMANDES) -> tuple[Plan, ...]:
    """Tous les plans entiers qui servent chaque demande a temps, sans surplus."""
    total = sum(demandes)
    cumulees = tuple(itertools.accumulate(demandes))
    retenus = []
    for debut in itertools.product(range(total + 1), repeat=len(demandes) - 1):
        plan = (*debut, total - sum(debut))
        produit = itertools.accumulate(plan)
        if plan[-1] >= 0 and all(
            p >= c for p, c in zip(produit, cumulees, strict=True)
        ):
            retenus.append(plan)
    return tuple(retenus)


def stock(plan: Plan, demandes: Plan = DEMANDES) -> int:
    """Les unites en stock a la fin de chacun des quatre premiers mois, sommees."""
    produit = itertools.accumulate(plan[:-1])
    return sum(produit) - avances_cumulees(demandes)


def cout(variante: str, plan: Plan, demandes: Plan = DEMANDES) -> int:
    """Production, stockage sauf pour la variante I, et supplement de la variante."""
    production = sum(c * q for c, q in zip(COUTS, plan, strict=True))
    stockage = 0 if variante == "I" else stock(plan, demandes)
    supplement = SUPPLEMENTS[variante](plan) if variante in SUPPLEMENTS else 0
    return production + stockage + supplement


@cache
def _classement(variante: str, demandes: Plan) -> tuple[tuple[int, Plan], ...]:
    if variante not in VARIANTES:
        msg = f"variante inconnue : {variante}"
        raise KeyError(msg)
    permis = [
        p
        for p in plans(demandes)
        if variante not in CONDITIONS or CONDITIONS[variante](p)
    ]
    lignes = [(cout(variante, p, demandes), p) for p in permis]
    if not lignes:
        msg = f"aucun plan ne respecte la variante {variante}"
        raise ValueError(msg)
    return tuple(sorted(lignes))


def meilleur(variante: str) -> tuple[int, Plan]:
    """Le cout minimal de la variante, et le plan qui l'atteint."""
    return _classement(variante, DEMANDES)[0]


def ex_aequo(variante: str) -> int:
    """Le nombre de plans qui atteignent le cout minimal."""
    lignes = _classement(variante, DEMANDES)
    return sum(1 for valeur, _ in lignes if valeur == lignes[0][0])


def meilleur_pour(demandes: Plan) -> tuple[int, Plan]:
    """Variante XI : le meilleur plan, stockage compris, pour ces demandes."""
    return _classement("stock", demandes)[0]
