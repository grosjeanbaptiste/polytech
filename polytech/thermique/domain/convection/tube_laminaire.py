"""La convection forcee dans un tube, en ecoulement laminaire.

Syllabus de thermique, tome 2, premiere partie, section 3.3.1. Une
couche limite thermique part de l'entree et se referme sur l'axe a la
distance `L_e,th = 0,05 Re_d Pr d`, soit `1,7 Pr` fois la longueur
d'entree hydrodynamique. Le coefficient local, infini a l'entree,
decroit jusqu'a cette distance, puis ne varie plus : `Nu = 3,65` a
temperature de paroi constante.

Le Nusselt moyen d'un tube de longueur `L` depend donc du seul groupe
`Re_d Pr d / L`. Hausen en donne la theorie ; pour un tube court devant
sa longueur d'entree -- `Re_d Pr d / L > 10` -- Sieder et Tate
proposent une relation plus simple. L'une et l'autre corrigent l'effet
de la temperature sur la viscosite par `(mu_m / mu_p)^0,14`.
"""

from __future__ import annotations

from typing import Final

NUSSELT_ETABLI: Final = 3.65
"""`Nu` loin de l'entree, a temperature de paroi constante."""
GROUPE_MINIMAL: Final = 10.0
"""Sous cette valeur du groupe d'entree, Sieder et Tate ne valent plus."""
_TIERS = 1.0 / 3.0


def longueur_d_entree_thermique(
    diametre: float, reynolds: float, prandtl: float
) -> float:
    """`L_e,th = 0,05 Re_d Pr d`."""
    return 0.05 * reynolds * prandtl * diametre


def groupe_d_entree(
    reynolds: float, prandtl: float, diametre: float, longueur: float
) -> float:
    """`Re_d Pr d / L`."""
    if longueur <= 0:
        msg = f"longueur nulle ou negative : {longueur} m"
        raise ValueError(msg)
    return reynolds * prandtl * diametre / longueur


def nusselt_de_hausen(groupe: float, *, viscosites: float = 1.0) -> float:
    """Le Nusselt moyen ; `viscosites` est le rapport `mu_m / mu_p`."""
    entree = 0.0668 * groupe / (1 + 0.04 * groupe ** (2 / 3))
    return float((NUSSELT_ETABLI + entree) * viscosites**0.14)


def nusselt_de_sieder_et_tate(
    groupe: float, *, viscosites: float = 1.0, hors_validite: bool = False
) -> float:
    """`1,86 (Re_d Pr d / L)^1/3 (mu_m / mu_p)^0,14`.

    `hors_validite` leve la garde `Re Pr d / L >= 10` pour montrer ce que la
    relation donnerait au-dela de son domaine (courrier a l'enseignant).
    """
    if groupe < GROUPE_MINIMAL and not hors_validite:
        msg = f"tube trop long pour cette relation : Re Pr d / L = {groupe:.3g}"
        raise ValueError(msg)
    return float(1.86 * groupe**_TIERS * viscosites**0.14)
