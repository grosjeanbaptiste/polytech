"""La theorie de la lubrification : un patin incline sur un film d'huile.

Planches du chapitre VII. Deux plaques non paralleles sont separees par un
film newtonien ; la plaque inferieure glisse a la vitesse `U`. Quand le
film est mince, `Re_L h^2 / L^2 << 1`, Navier-Stokes se reduit aux
equations de Reynolds : `du/dx + dv/dy = 0` et `dp/dx = mu d2u/dy2`.

Le profil est la somme d'un ecoulement de Poiseuille et d'un ecoulement de
Couette, `u = -(dp/dx)(h^2 / 2 mu)(1 - y/h)(y/h) + U (1 - y/h)`. Le debit
etant le meme dans toutes les sections, la pression du film vaut

`p(h) - p_0 = 6 mu U L (h_0 - h)(h - h_L) / (h^2 (h_0^2 - h_L^2))`,

nulle aux deux bouts, maximale en `h = 2 h_0 h_L / (h_0 + h_L)`. C'est
elle qui porte la charge.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


def critere_de_lubrification(
    *,
    masse_volumique: float,
    viscosite: float,
    vitesse: float,
    longueur: float,
    epaisseur: float,
) -> float:
    """`Re_L h^2 / L^2`, qui doit etre petit devant un."""
    reynolds = masse_volumique * vitesse * longueur / viscosite
    return reynolds * (epaisseur / longueur) ** 2


def vitesse_dans_le_film(
    ordonnee: Grandeur,
    epaisseur: Grandeur,
    vitesse: Grandeur,
    viscosite: Grandeur,
    pente: Grandeur,
) -> sp.Expr:
    """`u(y)`, `pente` etant le gradient de pression `dp/dx`."""
    reduit = ordonnee / epaisseur
    poiseuille = -pente * epaisseur**2 / (2 * viscosite) * (1 - reduit) * reduit
    return poiseuille + vitesse * (1 - reduit)


def residu_de_reynolds(
    profil: sp.Expr, ordonnee: sp.Symbol, viscosite: Grandeur, pente: Grandeur
) -> sp.Expr:
    """`dp/dx - mu d2u/dy2` : nul pour le profil du film."""
    return sp.simplify(pente - viscosite * sp.diff(profil, ordonnee, 2))


@dataclass(frozen=True, slots=True)
class Patin:
    """La longueur, les epaisseurs d'entree et de sortie, la vitesse, la viscosite."""

    longueur: float
    entree: float
    sortie: float
    vitesse: float
    viscosite: float

    def __post_init__(self) -> None:
        """Refuse un film qui ne se resserre pas dans le sens du mouvement."""
        if not 0 < self.sortie < self.entree:
            msg = f"film non convergent : {self.entree} m a l'entree, {self.sortie} m"
            raise ValueError(msg)

    def debit(self) -> float:
        """`q = U h_0 h_L / (h_0 + h_L)`, par unite de largeur."""
        return self.vitesse * self.entree * self.sortie / (self.entree + self.sortie)

    def surpression(self, epaisseur: float) -> float:
        """`p(h) - p_0`, en pascals."""
        if not self.sortie <= epaisseur <= self.entree:
            msg = f"epaisseur hors du film : {epaisseur} m"
            raise ValueError(msg)
        portee = (self.entree - epaisseur) * (epaisseur - self.sortie)
        echelle = 6 * self.viscosite * self.vitesse * self.longueur
        return echelle * portee / (epaisseur**2 * (self.entree**2 - self.sortie**2))

    def surpression_imprimee_planche_12(self, epaisseur: float) -> float:
        """La formule de la planche 12 telle qu'imprimee : sans `h^2`.

        Elle vaut `p(h) h^2`, une pression fois une surface : le defaut
        releve dans le courrier a l'enseignant. Ne sert qu'a le montrer.
        """
        return self.surpression(epaisseur) * epaisseur**2

    def epaisseur_du_maximum(self) -> float:
        """`h = 2 h_0 h_L / (h_0 + h_L)`."""
        return 2 * self.entree * self.sortie / (self.entree + self.sortie)

    def pression_maximale(self) -> float:
        """`(3 mu U L / (2 h_0 h_L)) (h_0 - h_L) / (h_0 + h_L)`."""
        return self.surpression(self.epaisseur_du_maximum())
