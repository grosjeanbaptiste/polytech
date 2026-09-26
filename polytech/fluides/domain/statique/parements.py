"""Geometries de parement concretes (voir :mod:`.efforts`).

Chacune ne declare que sa largeur mouillee ; tout le reste est derive.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.fluides.domain.statique.efforts import Parement
from polytech.fluides.domain.statique.symboles import ALPHA, H, L, Z


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


@dataclass(frozen=True, slots=True)
class ParementIncline(Parement):
    """Plan immerge incline d'un angle ``theta`` sur la surface libre.

    Le cours demande d'exprimer la force sur une surface plane « placee a
    la verticale ou inclinee d'un angle theta ». L'integrale se menant sur
    la *profondeur*, un parement incline offre, par unite de profondeur,
    une paroi de largeur ``L / sin(theta)`` : plus il se couche, plus il
    expose de surface a la meme hauteur d'eau. Le cas vertical est celui
    ou ``theta`` vaut un quart de tour.
    """

    largeur: float = 1.0
    angle: sp.Expr = sp.pi / 2

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        if self.largeur <= 0:
            msg = f"largeur non strictement positive : {self.largeur}"
            raise ValueError(msg)
        # Un angle symbolique n'a pas de valeur de verite : on ne refuse
        # que ce qui est demontrablement invalide, jamais l'indetermine.
        if sp.sin(self.angle).is_positive is False:
            msg = f"angle hors de ]0, pi[ : {self.angle}"
            raise ValueError(msg)

    def largeur_mouillee(self) -> sp.Expr:
        """Largeur de paroi par unite de profondeur."""
        return L / sp.sin(self.angle)

    def exposant_volume(self) -> int:
        """Bassin prismatique : le volume varie comme la hauteur."""
        return 1

    def _valeurs(self) -> dict[sp.Symbol, float]:
        return {L: self.largeur}
