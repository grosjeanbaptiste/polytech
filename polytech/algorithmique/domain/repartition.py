"""Repartir les lignes d'un fichier entre deux conteneurs, puis les entrelacer.

L'examen du 13 janvier 2025 (« Parcours Vita 2025 ») : chaque ligne de
`Participant.txt` s'ecrit `Val;Prenom;Age`. « Si Val = 1, le participant
sera stocke dans la List L1 ; si Val = 2, dans le Vector V2 », par insertion
a la fin. Le choix 5 « entrelace les prenoms des participants de L1 et V2
en commencant par L1 » ; l'exemple imprime montre que le conteneur le plus
long garde sa fin telle quelle.
"""

from __future__ import annotations

from itertools import zip_longest
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_CONTENEURS: Final = (1, 2)
_CHAMPS: Final = 3
_ABSENT: Final = object()


def _participant(ligne: str) -> tuple[int, str, int]:
    champs = ligne.split(";")
    if len(champs) != _CHAMPS or not (champs[0].isdigit() and champs[2].isdigit()):
        msg = f"ligne mal formee, Val;Prenom;Age attendu : {ligne}"
        raise ValueError(msg)
    return int(champs[0]), champs[1], int(champs[2])


def partager(lignes: Iterable[str]) -> dict[int, list[tuple[str, int]]]:
    """Les participants de chaque conteneur, dans l'ordre du fichier."""
    partage: dict[int, list[tuple[str, int]]] = {}
    for ligne in lignes:
        val, prenom, age = _participant(ligne)
        if val not in _CONTENEURS:
            msg = f"Val vaut 1 ou 2, et non {val}"
            raise ValueError(msg)
        partage.setdefault(val, []).append((prenom, age))
    return partage


def entrelacer(premiere: Sequence[Any], seconde: Sequence[Any]) -> list[Any]:
    """Un element de l'une, un de l'autre, et la fin de la plus longue."""
    paires = zip_longest(premiere, seconde, fillvalue=_ABSENT)
    return [element for paire in paires for element in paire if element is not _ABSENT]
