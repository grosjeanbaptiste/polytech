"""Les chaines de caracteres : le tableau de `char`, et les conversions de `string`.

Le cours (partie 1, planches 71 a 76) : une chaine est soit un tableau de
`char` termine par le caractere nul, de code ASCII 0, soit un
`string`. Le tableau doit donc reserver une case de plus que la chaine n'a
de lettres ; sa longueur se compte jusqu'au caractere nul.

`string` permet ce que le tableau interdit -- affecter, concatener,
comparer -- et se convertit en nombre par `stoi` et `stof`, qui lisent le
plus long prefixe numerique apres les blancs de tete et levent
`invalid_argument` s'il n'y en a pas.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from polytech.algorithmique.domain.types_cpp import TYPES_ENTIERS

if TYPE_CHECKING:
    from collections.abc import Sequence

NUL: Final = "\0"
"""Le caractere nul qui termine un tableau de `char`."""

_ENTIER: Final = re.compile(r"\s*([+-]?\d+)")
_REEL: Final = re.compile(r"\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")


def tableau_de_char(texte: str, dimension: int | None = None) -> tuple[str, ...]:
    """Les cases de `char nom[dimension] = "texte";`, caractere nul compris."""
    cases = dimension if dimension is not None else len(texte) + 1
    if cases <= len(texte):
        msg = f"pas de place pour le caractere nul : {cases} cases pour {texte!r}"
        raise ValueError(msg)
    return tuple(texte) + (NUL,) * (cases - len(texte))


def longueur(tableau: Sequence[str]) -> int:
    """Le nombre de caracteres avant le caractere nul."""
    if NUL not in tableau:
        msg = "aucun caractere nul : la chaine n'est pas terminee"
        raise ValueError(msg)
    return list(tableau).index(NUL)


def sous_chaine(texte: str, debut: int, nombre: int) -> str:
    """`string(texte, debut, nombre)` : au plus `nombre` caracteres depuis `debut`."""
    if not 0 <= debut <= len(texte):
        msg = f"position hors de la chaine : {debut} pour {len(texte)} caracteres"
        raise IndexError(msg)
    return texte[debut : debut + nombre]


def _prefixe(motif: re.Pattern[str], texte: str) -> str:
    trouve = motif.match(texte)
    if trouve is None:
        msg = f"invalid_argument : aucun nombre en tete de {texte!r}"
        raise ValueError(msg)
    return trouve.group(1)


def stoi(texte: str) -> int:
    """L'entier que porte le debut de la chaine."""
    valeur = int(_prefixe(_ENTIER, texte))
    if not TYPES_ENTIERS["int"].contient(valeur):
        msg = f"out_of_range : {valeur} ne tient pas dans un int"
        raise OverflowError(msg)
    return valeur


def stof(texte: str) -> float:
    """Le reel que porte le debut de la chaine."""
    return float(_prefixe(_REEL, texte))


NPOS: Final = 2**64 - 1
"""`string::npos` : ce que rend une recherche infructueuse, le plus grand `size_t`."""


def find(texte: str, motif: str) -> int:
    """`texte.find(motif)` : l'indice de la premiere occurrence, ou `NPOS`."""
    indice = texte.find(motif)
    return NPOS if indice < 0 else indice


def insert(texte: str, position: int, ajout: str) -> str:
    """`texte.insert(position, ajout)` : l'ajout se place devant ce rang."""
    if not 0 <= position <= len(texte):
        msg = f"position hors de la chaine : {position} pour {len(texte)} caracteres"
        raise IndexError(msg)
    return texte[:position] + ajout + texte[position:]


def precede(avant: str, apres: str) -> bool:
    """Vrai si `avant` precede `apres` dans l'ordre lexicographique des codes.

    La definition recurrente du syllabus : la chaine vide precede toute
    autre ; sinon on compare les premiers caracteres, et a egalite les
    suites.
    """
    if not avant or not apres:
        return bool(apres)
    if avant[0] != apres[0]:
        return ord(avant[0]) < ord(apres[0])
    return precede(avant[1:], apres[1:])
