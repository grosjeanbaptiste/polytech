"""Le corps noir : Stefan-Boltzmann, Wien, et la loi de Planck.

Le formulaire donne ces trois-la sous forme de courbes -- la loi de Planck
tracee pour 300 K et 6000 K, et la loi integree en abaque de fraction. Les
expressions analytiques sont donc etablies ici, et chacune se verifie
contre les deux autres : l'integrale de Planck sur tout le spectre doit
redonner Stefan-Boltzmann, et son maximum doit tomber sur Wien. Ce sont
ces deux recoupements que les tests exigent.
"""

from __future__ import annotations

import math
from typing import Final

STEFAN_BOLTZMANN: Final = 5.670374419e-8
"""Constante de Stefan-Boltzmann, en W/(m^2 K^4)."""

WIEN: Final = 2.897771955e-3
"""Constante de deplacement de Wien, en m K."""

PLANCK: Final = 6.62607015e-34
"""Constante de Planck, en J s."""

BOLTZMANN: Final = 1.380649e-23
"""Constante de Boltzmann, en J/K."""

LUMIERE: Final = 2.99792458e8
"""Vitesse de la lumiere dans le vide, en m/s."""


def emittance(temperature: float) -> float:
    """`sigma T^4`, l'emittance totale du corps noir, en W/m^2."""
    if temperature < 0:
        msg = f"temperature absolue negative interdite : {temperature} K"
        raise ValueError(msg)
    return STEFAN_BOLTZMANN * temperature**4


def longueur_d_onde_du_maximum(temperature: float) -> float:
    """Loi de Wien : `lambda_max = 2898 micrometres K / T`, en metres."""
    if temperature <= 0:
        msg = f"temperature nulle ou negative : {temperature} K"
        raise ValueError(msg)
    return WIEN / temperature


def luminance_spectrale(longueur_d_onde: float, temperature: float) -> float:
    """Loi de Planck, en W/(m^2 m) -- emittance par unite de longueur d'onde."""
    if longueur_d_onde <= 0:
        msg = f"longueur d'onde nulle ou negative : {longueur_d_onde} m"
        raise ValueError(msg)
    numerateur = 2.0 * math.pi * PLANCK * LUMIERE**2
    exposant = PLANCK * LUMIERE / (longueur_d_onde * BOLTZMANN * temperature)
    return numerateur / (longueur_d_onde**5 * (math.expm1(exposant)))


def absorptivite_de_kirchhoff(emissivite: float) -> float:
    """`alpha = epsilon` a l'equilibre thermique : la loi de Kirchhoff.

    Elle ferme le chapitre. Sans elle, un corps gris aurait deux
    proprietes radiatives independantes, et aucun bilan ne se fermerait --
    on ne saurait pas combien il absorbe de ce qu'il recoit.
    """
    if not 0 <= emissivite <= 1:
        msg = f"emissivite hors de [0, 1] : {emissivite}"
        raise ValueError(msg)
    return emissivite


def fraction_emise_sous(longueur_d_onde: float, temperature: float) -> float:
    """La part de l'emittance totale emise en dessous de cette longueur d'onde.

    C'est la « loi integree » que le formulaire donne en abaque -- un
    filtre passe-bas, en fonction du seul produit `lambda T`. Elle est
    calculee ici par la serie de Planck integree, et ses deux bornes sont
    exactes : zero en zero, un a l'infini.

    Son usage principal est l'effet de serre : a 5800 K le soleil emet
    plus de 95 pour cent sous trois micrometres, longueur a laquelle le
    verre est transparent, alors qu'il est opaque au rayonnement bien plus
    long que la piece reemet.
    """
    if longueur_d_onde <= 0:
        msg = f"longueur d'onde nulle ou negative : {longueur_d_onde} m"
        raise ValueError(msg)
    if temperature <= 0:
        msg = f"temperature nulle ou negative : {temperature} K"
        raise ValueError(msg)
    reduit = PLANCK * LUMIERE / (BOLTZMANN * longueur_d_onde * temperature)
    if reduit > _REDUIT_MAXIMAL:
        return 0.0
    somme = 0.0
    for terme in range(1, _TERMES_DE_LA_SERIE):
        produit = terme * reduit
        somme += (
            math.exp(-produit)
            * (produit**3 + 3 * produit**2 + 6 * produit + 6)
            / terme**4
        )
    return min(1.0, _FACTEUR_DE_LA_SERIE * somme)


_TERMES_DE_LA_SERIE = 60
"""Au-dela, les termes sont sous la precision du flottant."""

_REDUIT_MAXIMAL = 700.0
"""Au-dela, `exp(-x)` vaut zero en flottant : la fraction est nulle."""

_FACTEUR_DE_LA_SERIE = 15.0 / math.pi**4
"""Normalisation de la serie : elle doit tendre vers un."""
