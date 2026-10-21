"""Les pertes de charge en conduite : paroi, frottement, singularites.

Chapitre V du cours, planches 8 a 26 de
`MecaniqueFluides/SLIDES/Slides_04_EcoulementConduite_Similitudes.pdf` ;
`polytech.fluides.domain.conduite` en porte Poiseuille et Darcy-Weisbach.

* **Diametre hydraulique** (planche 9) : `D_h = 4 A / P`, le diametre d'une
  conduite non circulaire dans le nombre de Reynolds.
* **Paroi** (planches 11 et 19) : l'equilibre d'une tranche donne
  `Delta p = 4 L tau_w / D`, et Poiseuille `tau_w = 4 mu V_moy / R`.
* **Frottement turbulent** (planches 21 a 23) : Colebrook,
  `1/sqrt(l) = -2 log10(eps / (3,71 D) + 2,51 / (Re sqrt(l)))`, que le
  diagramme de Moody trace, et sa limite rugueuse de Nikuradse,
  `1/sqrt(l) = 1,14 - 0,868 ln(eps / D)`.
* **Singularites** (planches 24 a 26) : `Delta P = xi rho V^2 / 2`, et le
  total somme pertes reparties et singulieres.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.fluides.domain.regime_de_reynolds import REYNOLDS_TURBULENT

if TYPE_CHECKING:
    from collections.abc import Iterable

ENTREE_A_BORD_VIF: Final = 0.50
"""Planche 24 : `K_L` d'une entree a bord vif (recirculation)."""
ENTREE_A_BORD_ARRONDI: Final = 0.03
"""Planche 24 : `K_L` d'une entree bien arrondie."""
_TOLERANCE: Final = 1e-14


def diametre_hydraulique(aire: float, perimetre: float) -> float:
    """`D_h = 4 A / P` ; leve sans section ou sans perimetre."""
    if aire <= 0 or perimetre <= 0:
        msg = f"section sans aire ou sans perimetre : A = {aire}, P = {perimetre}"
        raise ValueError(msg)
    return 4 * aire / perimetre


def vitesse_moyenne(profil: sp.Expr, r: sp.Symbol, rayon: sp.Expr) -> sp.Expr:
    """`V_moy = (2 / R^2) int_0^R u(r) r dr`, en conduite cylindrique."""
    return sp.simplify(2 / rayon**2 * sp.integrate(profil * r, (r, 0, rayon)))


def contrainte_parietale_laminaire(
    viscosite: float, vitesse: float, rayon: float
) -> float:
    """`tau_w = -mu du/dr (R) = 4 mu V_moy / R`, en regime de Poiseuille."""
    return 4 * viscosite * vitesse / rayon


def chute_de_pression_par_la_paroi(
    contrainte: float, longueur: float, diametre: float
) -> float:
    """`Delta p = 4 L tau_w / D` : la paroi retient ce que la pression pousse."""
    return 4 * longueur * contrainte / diametre


def coefficient_de_nikuradse(rugosite_relative: float) -> float:
    """`1/sqrt(lambda) = 1,14 - 0,868 ln(eps / D)`, paroi rugueuse."""
    if rugosite_relative <= 0:
        msg = f"Nikuradse demande une rugosite : eps / D = {rugosite_relative}"
        raise ValueError(msg)
    return float((1.14 - 0.868 * math.log(rugosite_relative)) ** -2)


def coefficient_de_colebrook(reynolds: float, rugosite_relative: float) -> float:
    """Colebrook, resolue par point fixe sur `x = 1 / sqrt(lambda)`."""
    if reynolds <= REYNOLDS_TURBULENT or rugosite_relative < 0:
        msg = f"Colebrook hors domaine : Re = {reynolds}, eps / D = {rugosite_relative}"
        raise ValueError(msg)
    x = 1.0 / math.sqrt(0.02)
    while True:
        suivant = -2 * math.log10(rugosite_relative / 3.71 + 2.51 * x / reynolds)
        if abs(suivant - x) < _TOLERANCE * suivant:
            return float(suivant**-2)
        x = suivant


def perte_de_charge_singuliere(
    coefficient: float, masse_volumique: float, vitesse: float
) -> float:
    """`Delta P = xi rho V^2 / 2`, en pascals."""
    if coefficient < 0:
        msg = f"coefficient singulier negatif : xi = {coefficient}"
        raise ValueError(msg)
    return coefficient * masse_volumique * vitesse**2 / 2


def pertes_de_charge_totales(
    regulieres: Iterable[float], singulieres: Iterable[float]
) -> float:
    """Planche 26 : la somme des pertes reparties et singulieres."""
    pertes = [*regulieres, *singulieres]
    if any(p < 0 for p in pertes):
        msg = f"une perte de charge est positive : {pertes}"
        raise ValueError(msg)
    return float(sum(pertes))
