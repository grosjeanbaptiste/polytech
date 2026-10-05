"""La convection naturelle autour des corps immerges, et la lame de fluide.

Syllabus de thermique, tome 2, premiere partie, sections 4.1 a 4.3. Le
fluide se met en mouvement de lui-meme, sous les forces d'Archimede :
`Nu = F(Gr, Pr)`, et le plus souvent du seul produit `Gr Pr`.

Le long d'une paroi verticale, la couche limite reste laminaire jusqu'a
`Gr = 1e9`. Le Nusselt moyen y vaut `0,56 (Gr Pr)^1/4`, puis
`0,12 (Gr Pr)^1/3` : en laminaire le coefficient local varie comme
`z^-1/4`, et vaut donc les trois quarts du coefficient moyen.

Autour d'un cylindre horizontal, `0,47 (Gr Pr)^1/4`, puis
`0,10 (Gr Pr)^1/3` pour les gaz. Au-dessus d'une plaque horizontale,
`0,137 (Gr Pr)^1/3` : l'exposant un tiers rend le coefficient
independant de toute dimension.

Une lame de fluide entre deux plaques transmet comme un solide de
conductivite `lambda_eq = lambda F(Gr_b Pr)`.
"""

from __future__ import annotations

from typing import Final

from polytech.fluides.domain.grandeurs import PESANTEUR
from polytech.thermique.domain.convection.naturelle import (
    nusselt_du_cylindre_horizontal,
    nusselt_lame_de_fluide,
)

GRASHOF_CRITIQUE: Final = 1e9
"""(4.1) : au-dela, la couche limite devient turbulente."""
GRASHOF_MINIMAL: Final = 1e3
"""Sous cette valeur, le syllabus ne donne aucune relation."""
_TIERS = 1.0 / 3.0


def _exiger_le_domaine(nombre: float) -> None:
    if nombre < GRASHOF_MINIMAL:
        msg = f"hors du domaine des relations : {nombre:.3g} < {GRASHOF_MINIMAL:.0e}"
        raise ValueError(msg)


def hauteur_critique(dilatation: float, ecart: float, viscosite: float) -> float:
    """`z_crit`, telle que `beta g theta z^3 / nu^2 = 1e9`."""
    poussee = dilatation * PESANTEUR * abs(ecart)
    return float((GRASHOF_CRITIQUE * viscosite**2 / poussee) ** _TIERS)


def nusselt_de_la_paroi_verticale(rayleigh: float) -> float:
    """`0,56 (Gr Pr)^1/4` jusqu'a 1e9, `0,12 (Gr Pr)^1/3` au-dela."""
    _exiger_le_domaine(rayleigh)
    if rayleigh > GRASHOF_CRITIQUE:
        return float(0.12 * rayleigh**_TIERS)
    return float(0.56 * rayleigh**0.25)


def nusselt_du_cylindre(grashof: float, prandtl: float) -> float:
    """`0,47 (Gr Pr)^1/4` jusqu'a `Gr = 1e9`, `0,10 (Gr Pr)^1/3` au-dela."""
    _exiger_le_domaine(grashof)
    if grashof > GRASHOF_CRITIQUE:
        return float(0.10 * (grashof * prandtl) ** _TIERS)
    return nusselt_du_cylindre_horizontal(grashof * prandtl)


def nusselt_de_la_plaque_horizontale(rayleigh: float) -> float:
    """`0,137 (Gr Pr)^1/3`, la face chaude vers le haut."""
    return float(0.137 * rayleigh**_TIERS)


def epaisseur_d_eckert(hauteur: float, grashof: float, prandtl: float) -> float:
    """`delta = 3,93 z Pr^-1/2 (0,952 + Pr)^1/4 Gr^-1/4`, en laminaire."""
    forme = prandtl**-0.5 * (0.952 + prandtl) ** 0.25
    return float(3.93 * hauteur * forme * grashof**-0.25)


def coefficient_local_laminaire(moyen: float) -> float:
    """`K(z) = (3 / 4) K_moyen`, au sommet de la hauteur consideree."""
    return 0.75 * moyen


def coefficient_du_cylindre_dans_l_air(ecart: float, diametre: float) -> float:
    """`K = 1,3 (theta_o / d)^0,25`, pour l'air aux conditions ambiantes."""
    if diametre <= 0:
        msg = f"diametre nul ou negatif : {diametre} m"
        raise ValueError(msg)
    return float(1.3 * (abs(ecart) / diametre) ** 0.25)


def conductance_de_la_lame(
    rayleigh: float, conductivite: float, epaisseur: float
) -> float:
    """`lambda_eq / b`, en W/(m^2 K)."""
    return nusselt_lame_de_fluide(rayleigh) * conductivite / epaisseur
