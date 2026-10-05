"""Decider si une partie d'un espace vectoriel en est un sous-espace.

Le syllabus d'algebre (VII.1, exemple 1) deroule trois verifications, et
s'arrete a la premiere qui echoue : le vecteur nul appartient-il a la
partie ? est-elle stable pour la somme ? pour la multiplication par un
scalaire ?

Un « non » se prouve par un contre-exemple : `contre_exemple` le cherche
parmi des candidats, dans l'ordre des trois verifications. N'en trouver
aucun ne prouve rien. Un « oui » demande un argument : une partie definie
par des equations lineaires homogenes est le noyau d'une matrice, donc un
sous-espace -- ce que dit `est_lineaire_homogene`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING, Any

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(frozen=True, slots=True)
class Violation:
    """La verification qui echoue, et les temoins qui le montrent."""

    nature: str
    temoins: tuple[Any, ...]


def est_lineaire_homogene(
    equations: Sequence[sp.Expr], variables: Sequence[sp.Symbol]
) -> bool:
    """Vrai si chaque equation `e = 0` est du premier degre, sans terme constant."""
    for equation in equations:
        try:
            polynome = sp.Poly(equation, *variables)
        except sp.PolynomialError:
            return False
        constant = polynome.eval(dict.fromkeys(variables, 0))
        if polynome.total_degree() > 1 or constant != 0:
            return False
    return True


def contre_exemple(
    appartient: Callable[[Any], bool],
    candidats: Sequence[Any],
    scalaires: Sequence[object] = (-1, 2),
) -> Violation | None:
    """La premiere des trois verifications qui echoue sur les candidats, ou rien."""
    if not candidats:
        msg = "aucun candidat : rien ne peut servir de contre-exemple"
        raise ValueError(msg)
    nul = 0 * candidats[0]
    if not appartient(nul):
        return Violation("vecteur nul", (nul,))
    dedans = [candidat for candidat in candidats if appartient(candidat)]
    for premier, second in combinations_with_replacement(dedans, 2):
        if not appartient(premier + second):
            return Violation("somme", (premier, second))
    for scalaire in scalaires:
        for vecteur in dedans:
            if not appartient(scalaire * vecteur):
                return Violation("multiple", (scalaire, vecteur))
    return None
