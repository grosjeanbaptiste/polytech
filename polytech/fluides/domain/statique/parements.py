"""Geometries de parement concretes (voir :mod:`.efforts`).

Chacune ne declare que sa largeur mouillee ; tout le reste est derive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polytech.fluides.domain.statique.efforts import Parement
from polytech.fluides.domain.statique.symboles import ALPHA, H, L, Z

if TYPE_CHECKING:
    import sympy as sp


@dataclass(frozen=True, slots=True)
class ParementVertical(Parement):
    """Mur vertical de largeur constante ; bassin prismatique (``V = A H``)."""

    largeur: float = 1.0

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        if self.largeur <= 0:
            msg = f"largeur non strictement positive : {self.largeur}"
            raise ValueError(msg)

    def largeur_mouillee(self) -> sp.Expr:
        """Largeur constante ``L``, independante de la profondeur."""
        return L

    def exposant_volume(self) -> int:
        """Bassin prismatique : ``V = A H`` varie comme ``H``."""
        return 1

    def _valeurs(self) -> dict[sp.Symbol, float]:
        return {L: self.largeur}


@dataclass(frozen=True, slots=True)
class ParementTriangulaire(Parement):
    """Vallee en V auto-similaire : largeur ``2 alpha (H - z)``.

    Le volume y varie lui aussi comme ``H^3`` : c'est la seule geometrie ou
    poussee et volume sont proportionnels.
    """

    pente: float = 1.0

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        if self.pente <= 0:
            msg = f"pente non strictement positive : {self.pente}"
            raise ValueError(msg)

    def largeur_mouillee(self) -> sp.Expr:
        """Largeur maximale en surface, nulle au fond."""
        return 2 * ALPHA * (H - Z)

    def exposant_volume(self) -> int:
        """Vallee auto-similaire : le volume varie comme ``H^3``."""
        return 3

    def _valeurs(self) -> dict[sp.Symbol, float]:
        return {ALPHA: self.pente}
