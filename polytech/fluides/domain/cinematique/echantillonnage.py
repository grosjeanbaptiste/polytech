"""Echantillonnage d'un champ de vitesse pour le trace (cours III.3).

Le trace des lignes de courant se lit mieux avec le champ de vitesse
superpose. Ces fleches ne sont pas un ornement : elles disent le sens de
l'ecoulement, et elles doivent donc etre *calculees* a partir du champ --
non saisies a la main sous la figure, ou rien ne les verifierait.

Elles sont normalisees : a cette echelle, une fleche proportionnelle a la
vitesse serait illisible loin de l'origine. Seule leur direction porte du
sens, et la figure le dit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polytech.fluides.domain.cinematique.symboles import T, X, Y

if TYPE_CHECKING:
    from polytech.fluides.domain.cinematique.champ import ChampDeVitesse


@dataclass(frozen=True, slots=True)
class Fleche:
    """Un vecteur unitaire du champ, place en un noeud de la grille."""

    x: float
    y: float
    u: float
    v: float


@dataclass(frozen=True, slots=True)
class GrilleDeTrace:
    """Noeuds ou echantillonner le champ, et longueur des fleches."""

    abscisses: tuple[float, ...]
    ordonnees: tuple[float, ...]
    longueur: float

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        if not self.abscisses or not self.ordonnees:
            msg = "grille vide : aucun noeud a echantillonner"
            raise ValueError(msg)
        if self.longueur <= 0:
            msg = f"longueur de fleche non strictement positive : {self.longueur}"
            raise ValueError(msg)

    def noeuds(self) -> tuple[tuple[float, float], ...]:
        """Produit cartesien des abscisses et des ordonnees."""
        return tuple((x, y) for x in self.abscisses for y in self.ordonnees)


def fleches_du_champ(
    champ: ChampDeVitesse,
    grille: GrilleDeTrace,
    instant: float = 0.0,
) -> tuple[Fleche, ...]:
    """Evalue le champ sur la grille et normalise chaque vecteur."""
    return tuple(
        _fleche(champ, x, y, instant, grille.longueur) for x, y in grille.noeuds()
    )


def _fleche(
    champ: ChampDeVitesse,
    x: float,
    y: float,
    instant: float,
    longueur: float,
) -> Fleche:
    valeurs = {X: x, Y: y, T: instant}
    u = float(champ.u.subs(valeurs))
    v = float(champ.v.subs(valeurs))
    norme = math.hypot(u, v)
    if norme == 0:
        msg = f"vitesse nulle en ({x}, {y}) : sa direction n'est pas definie"
        raise ValueError(msg)
    facteur = longueur / norme
    return Fleche(x=x, y=y, u=u * facteur, v=v * facteur)
