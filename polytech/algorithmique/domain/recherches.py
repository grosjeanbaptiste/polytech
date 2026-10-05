"""Chercher un element dans un tableau : un par un, ou par moities.

Le cours (partie 1, planches 63, 68 et 69) :

* la **recherche lineaire** avance d'une case tant qu'elle n'a pas trouve
  et qu'il reste des cases : `while (i < N && A[i] != x) i = i + 1;`. Elle
  s'arrete sur le rang de l'element, ou sur `N` s'il est absent ;
* la **recherche dichotomique** ne vaut que pour un tableau **trie**. Elle
  compare l'element cherche a celui du milieu, `mid = (inf + sup) / 2`, et
  ne garde que la moitie ou il peut encore se trouver : l'intervalle est
  divise par deux a chaque etape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class Dichotomie:
    """Le rang trouve, ou `None`, et les milieux visites dans l'ordre."""

    position: int | None
    milieux: tuple[int, ...]


def recherche_lineaire(tableau: Sequence[Any], cherche: Any) -> int:  # noqa: ANN401 -- tout type
    """Le rang du premier element egal, ou la taille du tableau s'il est absent."""
    rang = 0
    while rang < len(tableau) and tableau[rang] != cherche:
        rang += 1
    return rang


def recherche_dichotomique(tableau: Sequence[Any], cherche: Any) -> Dichotomie:  # noqa: ANN401 -- tout type
    """La recherche par moities, avec la suite des milieux compares."""
    if any(tableau[i] > tableau[i + 1] for i in range(len(tableau) - 1)):
        msg = "tableau non trie : la recherche dichotomique n'y vaut pas"
        raise ValueError(msg)
    inferieur, superieur, milieux = 0, len(tableau) - 1, []
    while inferieur <= superieur:
        milieu = (inferieur + superieur) // 2
        milieux.append(milieu)
        if tableau[milieu] == cherche:
            return Dichotomie(milieu, tuple(milieux))
        if tableau[milieu] > cherche:
            superieur = milieu - 1
        else:
            inferieur = milieu + 1
    return Dichotomie(None, tuple(milieux))
