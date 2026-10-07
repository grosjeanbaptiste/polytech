"""Les structures, et les operations du TP 3 sur un tableau d'enregistrements.

Le cours (partie 1, planches 143 a 158 ; S3, planches 11 a 38) : une
structure reunit plusieurs variables sous un seul type. Elle se passe a
une fonction comme toute variable -- par **valeur**, et la fonction n'en
modifie qu'une copie, ou par reference -- et s'affecte d'un seul geste :
`Tab[i] = Tab[i+1]` recopie tous les champs.

Le TP 3 traite un tableau d'au plus dix articles, un nom et un code :
permuter deux a deux, trier selon le code ou selon le nom, chercher la
position d'un article, effacer les doublons.

Effacer un doublon decale les suivants : l'element qui prend sa place doit
etre examine a son tour, sans quoi deux doublons qui se suivent laissent
passer le second.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(frozen=True, slots=True)
class Article:
    """`struct article { string Nom; char Code; };`."""

    nom: str
    code: str


def permuter_deux_a_deux(elements: Sequence[Any]) -> tuple[Any, ...]:
    """Le premier avec le dernier, le deuxieme avec l'avant-dernier, et ainsi de suite.

    Les transparents ne disent pas ce que « deux a deux » veut dire ; la
    feuille de TP de 2019 le dit, page 7. On s'arrete a la moitie : aller
    plus loin referait chaque echange et remettrait tout en place.
    """
    resultat = list(elements)
    for rang in range(len(resultat) // 2):
        oppose = len(resultat) - 1 - rang
        resultat[rang], resultat[oppose] = resultat[oppose], resultat[rang]
    return tuple(resultat)


def trier_selon(elements: Sequence[Any], cle: Callable[[Any], Any]) -> tuple[Any, ...]:
    """Un tri a bulles sur la cle ; deux elements de meme cle gardent leur ordre."""
    resultat, permute = list(elements), True
    while permute:
        permute = False
        for rang in range(len(resultat) - 1):
            if cle(resultat[rang]) > cle(resultat[rang + 1]):
                resultat[rang], resultat[rang + 1] = resultat[rang + 1], resultat[rang]
                permute = True
    return tuple(resultat)


def position_de(articles: Sequence[Article], nom: str, code: str) -> int:
    """Le rang du premier article de ce nom et de ce code, ou la taille du tableau."""
    rang = 0
    while rang < len(articles) and articles[rang] != Article(nom, code):
        rang += 1
    return rang


def sans_doublons(elements: Sequence[Any]) -> tuple[Any, ...]:
    """Chaque element garde, a sa premiere occurrence seulement."""
    gardes: list[Any] = []
    for element in elements:
        if element not in gardes:
            gardes.append(element)
    return tuple(gardes)
