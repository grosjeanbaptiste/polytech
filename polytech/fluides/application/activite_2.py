"""L'activite d'integration 2 : film d'huile, reservoir perce, bateau-pompe.

Les trois exercices sont ceux du chapitre IV du cours, et le domaine les
porte deja (`film_tombant`, `remplissage`, `bernoulli_generalise`). Ce
module ne fait que les appliquer aux donnees de l'enonce
(`MecaniqueFluides/EXS/Exs_Activite_integration_2.pdf`) et preparer les
recoupements que le corrige publie.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.fluides.domain.fondamentales.bernoulli import GRAVITE_EXACTE
from polytech.fluides.domain.fondamentales.bernoulli_generalise import (
    hauteur_manometrique,
    puissance_a_l_arbre_d_une_pompe,
    puissance_hydraulique,
)
from polytech.fluides.domain.fondamentales.continuite import vitesse_dans_la_section
from polytech.fluides.domain.fondamentales.film_tombant import vitesse_du_film
from polytech.fluides.domain.fondamentales.remplissage import duree_de_remplissage

X, H, RHO, MU, G = sp.symbols("x h rho mu g", positive=True)
ABSCISSE_REDUITE: Final = sp.Symbol("x^*", positive=True)
"""`x* = x / h`, de la paroi (0) a la surface libre (1)."""

EAU_DE_MER: Final = 1030.0
DEBIT: Final = 0.1
DIAMETRE_D_ASPIRATION: Final = 0.20
DIAMETRE_DE_LA_BUSE: Final = 0.05
PERTES: Final = 3.0
HAUTEUR_DE_LA_BUSE: Final = 4.0
RENDEMENT: Final = 0.70
"""Les donnees du bateau-pompe, en unites SI, telles que l'enonce les ecrit."""


@dataclass(frozen=True, slots=True)
class BateauPompe:
    """Ce que l'exercice demande, et les etapes qui y menent."""

    aspiration: float
    ejection: float
    hauteur: float
    hydraulique: float
    arbre: float


def profil_normalise() -> sp.Expr:
    """`w* = w mu / (rho g h^2)` en fonction de `x* = x / h`."""
    reduit = (
        vitesse_du_film(ABSCISSE_REDUITE * H, H, RHO, MU, G) * MU / (RHO * G * H**2)
    )
    return sp.factor(sp.simplify(reduit))


def profil_par_integration() -> sp.Expr:
    """`mu w'' = rho g`, integree deux fois avec `w(0) = 0` et `w'(h) = 0`."""
    w = sp.Function("w")
    equation = sp.Eq(MU * w(X).diff(X, 2), RHO * G)
    conditions = {w(0): 0, w(X).diff(X).subs(X, H): 0}
    return sp.factor(sp.dsolve(equation, w(X), ics=conditions).rhs)


def cisaillement_a_la_paroi() -> sp.Expr:
    """`tau = mu dw/dx` en `x = 0` : ce que la paroi retient."""
    return sp.simplify(MU * sp.diff(vitesse_du_film(X, H, RHO, MU, G), X).subs(X, 0))


def rapport_moyenne_sur_surface() -> sp.Expr:
    """La vitesse moyenne du film, rapportee a celle de sa surface libre."""
    profil = vitesse_du_film(X, H, RHO, MU, G)
    moyenne = sp.integrate(profil, (X, 0, H)) / H
    return sp.simplify(moyenne / profil.subs(X, H))


def hauteur_maximale(debit_massique: sp.Expr, orifice: sp.Expr) -> sp.Expr:
    """`z_max = (m / (rho A_o))^2 / (2 g)` : l'orifice evacue tout ce qui entre."""
    return (debit_massique / (RHO * orifice)) ** 2 / (2 * G)


def duree_reduite(fraction: float) -> sp.Expr:
    """`t / T` pour atteindre cette fraction de `z_max`, `T = 2 rho A_T z_max / m`.

    Calculee sur un reservoir unite -- debit, masse volumique et sections
    egaux a un -- ou `z_max = 1 / (2 g)` et `T = 1 / g`.
    """
    un = sp.Integer(1)
    niveau = sp.nsimplify(fraction) / (2 * GRAVITE_EXACTE)
    return sp.simplify(duree_de_remplissage(niveau, un, un, un, un) * GRAVITE_EXACTE)


def bateau_pompe() -> BateauPompe:
    """De la surface de la mer, au repos, au jet a l'air libre."""
    section = math.pi * DIAMETRE_DE_LA_BUSE**2 / 4
    ejection = vitesse_dans_la_section(DEBIT, section)
    aspiration = vitesse_dans_la_section(DEBIT, math.pi * DIAMETRE_D_ASPIRATION**2 / 4)
    hauteur = hauteur_manometrique(
        denivele=HAUTEUR_DE_LA_BUSE, vitesse_finale=ejection, pertes=PERTES
    )
    hydraulique = puissance_hydraulique(EAU_DE_MER, DEBIT, hauteur)
    arbre = puissance_a_l_arbre_d_une_pompe(
        EAU_DE_MER, DEBIT, hauteur, rendement_pompe=RENDEMENT
    )
    return BateauPompe(aspiration, ejection, hauteur, hydraulique, arbre)
