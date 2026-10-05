"""L'interpolation lineaire dans une table de proprietes.

Les tables du syllabus donnent les proprietes a des temperatures
espacees ; entre deux lignes, chaque colonne s'interpole lineairement.
Hors de la table, rien n'est rendu : une propriete extrapolee serait
une valeur que personne n'a mesuree.
"""

from __future__ import annotations

from dataclasses import Field, fields
from itertools import pairwise
from typing import TYPE_CHECKING, ClassVar, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence


class Ligne(Protocol):
    """Une ligne de table : une classe de donnees qui porte sa temperature."""

    __dataclass_fields__: ClassVar[dict[str, Field[object]]]

    @property
    def temperature(self) -> float:
        """L'abscisse de la table, dans l'unite que celle-ci emploie."""


def _entre[L: Ligne](bas: L, haut: L, temperature: float) -> L:
    part = (temperature - bas.temperature) / (haut.temperature - bas.temperature)
    colonnes = {
        champ.name: getattr(bas, champ.name)
        + part * (getattr(haut, champ.name) - getattr(bas, champ.name))
        for champ in fields(bas)
    }
    return type(bas)(**colonnes)


def interpoler[L: Ligne](table: Sequence[L], temperature: float) -> L:
    """La ligne a cette temperature, chaque colonne interpolee."""
    if not table:
        msg = "table vide : rien a interpoler"
        raise ValueError(msg)
    for bas, haut in pairwise(table):
        if bas.temperature <= temperature <= haut.temperature:
            return _entre(bas, haut, temperature)
    bornes = (table[0].temperature, table[-1].temperature)
    msg = f"temperature hors de la table {bornes} : {temperature}"
    raise ValueError(msg)
