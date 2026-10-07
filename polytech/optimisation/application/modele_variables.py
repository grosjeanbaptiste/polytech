"""Des variables nommees, pour ecrire un modele comme les planches l'ecrivent.

Un programme lineaire range ses coefficients par position. Un modele de
cinquante variables ecrit ainsi ne se relit pas : on les nomme donc, et
une ligne se donne par ses seuls termes non nuls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import sympy as sp


@dataclass(frozen=True, slots=True)
class Variables:
    """Les noms des variables d'un modele, dans l'ordre de leurs colonnes."""

    noms: tuple[str, ...]

    def ligne(self, termes: Mapping[str, sp.Expr | int]) -> tuple[sp.Expr | int, ...]:
        """La ligne de coefficients qui porte ces termes, et zero ailleurs."""
        inconnus = sorted(set(termes) - set(self.noms))
        if inconnus:
            msg = f"variable inconnue : {inconnus[0]}"
            raise KeyError(msg)
        return tuple(termes.get(nom, 0) for nom in self.noms)

    def lire(self, point: Sequence[sp.Expr], prefixe: str) -> tuple[sp.Expr, ...]:
        """Les valeurs, dans un point, des variables dont le nom commence ainsi."""
        return tuple(
            valeur
            for nom, valeur in zip(self.noms, point, strict=True)
            if nom.startswith(prefixe)
        )


def numerotees(prefixe: str, nombre: int) -> tuple[str, ...]:
    """`T1`, `T2`, ... : une famille de variables indexee a partir de 1."""
    return tuple(f"{prefixe}{rang}" for rang in range(1, nombre + 1))
