"""Activite d'integration 2 : les hypotheses du corrige, mesurees.

Les proprietes viennent du cours, non d'une table exterieure :

- l'huile, `MecaniqueFluides/SLIDES/Slides_07_Lubrification_Ecoulement_turbulent.pdf`,
  planche 14 : « Huile SAE50 a 50 degres », rho = 900 kg/m3, mu = 0,1 N s/m2 ;
- l'air, `Slides_01_Introduction_2025_2026.pdf` : mu = 1,83 . 10^-5 Pa s a
  vingt degres et une atmosphere ; sa masse volumique sort du gaz parfait.

Le Reynolds du film est celui du cours, `rho V L / mu`, avec la vitesse
moyenne du film et son epaisseur : `Re = rho^2 g h^3 / (3 mu^2)`.
"""

from __future__ import annotations

import math
from typing import Final

from polytech.fluides.application import activite_2 as a2
from polytech.fluides.domain.grandeurs import PESANTEUR, Pression
from polytech.fluides.domain.statique.gaz import AIR

HUILE_MASSE_VOLUMIQUE: Final = 900.0
HUILE_VISCOSITE: Final = 0.1
AIR_VISCOSITE: Final = 1.83e-5
AIR_PRESSION: Final = 101325.0
AIR_TEMPERATURE: Final = 293.15
"""Vingt degres : la temperature ou le cours donne la viscosite de l'air."""


def rapport_des_viscosites() -> float:
    """`mu_air / mu_huile` : ce que la condition de surface libre neglige."""
    return AIR_VISCOSITE / HUILE_VISCOSITE


def masse_volumique_de_l_air() -> float:
    """`rho = p M / (R T)` a une atmosphere et vingt degres."""
    return AIR.masse_volumique(
        Pression(pascals=AIR_PRESSION), AIR_TEMPERATURE
    ).kg_par_m3


def rapport_des_masses_volumiques() -> float:
    """`rho_air / rho_huile` : la poussee de l'air allege la pesanteur d'autant."""
    return masse_volumique_de_l_air() / HUILE_MASSE_VOLUMIQUE


def reynolds_du_film(epaisseur: float) -> float:
    """`Re = rho V_moy h / mu`, avec `V_moy = rho g h^2 / (3 mu)`."""
    if epaisseur <= 0:
        msg = f"epaisseur de film non strictement positive : {epaisseur} m"
        raise ValueError(msg)
    rho, mu = HUILE_MASSE_VOLUMIQUE, HUILE_VISCOSITE
    return rho**2 * PESANTEUR * epaisseur**3 / (3 * mu**2)


def epaisseur_a_reynolds_unite() -> float:
    """L'epaisseur ou l'inertie et la viscosite sont du meme ordre."""
    rho, mu = HUILE_MASSE_VOLUMIQUE, HUILE_VISCOSITE
    return float(math.cbrt(3 * mu**2 / (rho**2 * PESANTEUR)))


def facteur_de_vitesse_de_surface(rapport: float) -> float:
    """`1 - (D_o / D_T)^4` : Bernoulli sans negliger la vitesse de la surface."""
    if not 0 < rapport < 1:
        msg = f"rapport des diametres hors de ]0, 1[ : {rapport}"
        raise ValueError(msg)
    return 1 - rapport**4


def facteur_de_perte(coefficient: float) -> float:
    """`1 + K` : une perte `K V^2 / (2 g)` a l'orifice eleve d'autant `z_max`."""
    if coefficient < 0:
        msg = f"coefficient de perte negatif : {coefficient}"
        raise ValueError(msg)
    return 1 + coefficient


def _puissance(debit: float, diametre: float) -> float:
    """La puissance a l'arbre, refaite pour un debit et une buse donnes."""
    cinetique = 8 * debit**2 / (math.pi**2 * PESANTEUR * diametre**4)
    hauteur = a2.HAUTEUR_DE_LA_BUSE + a2.PERTES + cinetique
    return a2.EAU_DE_MER * PESANTEUR * debit * hauteur / a2.RENDEMENT


def elasticites_du_bateau() -> dict[str, float]:
    """`d ln P / d ln X` pour chaque donnee : le poids de son incertitude."""
    bateau = a2.bateau_pompe()
    part = bateau.ejection**2 / (2 * PESANTEUR) / bateau.hauteur
    return {
        "diametre": -4 * part,
        "debit": 1 + 2 * part,
        "rendement": -1.0,
        "masse_volumique": 1.0,
    }


def ecarts_pour_un_millimetre_de_buse() -> tuple[float, float]:
    """L'ecart relatif de puissance pour une buse de 49 mm, puis de 51 mm."""
    reference = _puissance(a2.DEBIT, a2.DIAMETRE_DE_LA_BUSE)
    etroite = _puissance(a2.DEBIT, a2.DIAMETRE_DE_LA_BUSE - 1e-3) / reference - 1
    large = _puissance(a2.DEBIT, a2.DIAMETRE_DE_LA_BUSE + 1e-3) / reference - 1
    return etroite, large
