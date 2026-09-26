"""Geometries de parement concretes (voir :mod:`.efforts`).

Chacune ne declare que sa largeur mouillee ; tout le reste est derive.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.fluides.domain.statique.efforts import Parement
from polytech.fluides.domain.statique.symboles import (
    AIRE,
    ALLONGEMENT,
    ALPHA,
    H,
    L,
    Z,
)


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

    def section_horizontale(self) -> sp.Expr:
        """Berges verticales : l'aire de la surface libre ne varie pas."""
        return AIRE

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

    def section_horizontale(self) -> sp.Expr:
        """Vallee auto-similaire : largeur *et* longueur croissent avec ``H``.

        La surface libre mesure ``2 alpha H`` de large et ``lambda H`` de
        long : son aire varie donc comme ``H^2``, et le volume comme
        ``H^3``. C'est cette auto-similarite -- et elle seule -- qui rend
        poussee et volume proportionnels.
        """
        return 2 * ALPHA * ALLONGEMENT * H**2

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

    def section_horizontale(self) -> sp.Expr:
        """L'inclinaison change la paroi, pas la facon dont le bassin se remplit."""
        return AIRE

    def _valeurs(self) -> dict[sp.Symbol, float]:
        return {L: self.largeur}
