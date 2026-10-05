"""La conduction dans la matiere : lois d'Ohm et de Joule locales.

Le syllabus d'electricite generale (13.1 a 13.6). Les charges mobiles
d'un conducteur sont lentes et legeres : dans le cuivre, a 15 A/mm^2,
les electrons libres avancent d'un millimetre par seconde.

En negligeant leur inertie et en prenant la reaction du conducteur
proportionnelle a la vitesse relative, `f_ra = h_a (v_a - v)`, l'equilibre
de chaque espece donne la loi d'Ohm locale `E_s + E_e + E_d = rho i_cd`
(13.3.5), ou `gamma = 1 / rho = somme des eta_a^2 / h_a` (13.3.3). La
puissance des frottements est la loi de Joule, `p_j = rho i_cd^2` (13.4.2).

La charge volumique d'un conducteur suit (13.5.1) : elle est nulle s'il
est homogene et au repos. Le long de sa surface, le champ du dielectrique
garde la composante tangentielle `rho i_cd` et prend la normale
`sigma / epsilon` (13.6.4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.symboles import CARTESIENNES

if TYPE_CHECKING:
    from collections.abc import Sequence

    Triplet = tuple[Any, Any, Any]

CHARGE_LIBRE_DU_CUIVRE: Final = 13.6e9
"""En C/m^3 : les « 13,6 C/mm^3 » d'electrons libres de la page 13-2."""

NUCLEON_SUR_ELECTRON: Final = 1836
"""Le rapport des masses du proton, ou du neutron, a celle de l'electron."""


def vitesse_relative(courant: float, charge: float) -> float:
    """`v = i / eta`, en metres par seconde."""
    if charge == 0:
        msg = "charge volumique nulle : aucune charge mobile ne porte le courant"
        raise ValueError(msg)
    return courant / charge


def rapport_des_masses_volumiques(nombre_de_masse: int) -> int:
    """La matiere sur ses electrons libres, a un electron par atome."""
    return nombre_de_masse * NUCLEON_SUR_ELECTRON


def conductivite(especes: Sequence[tuple[float, float]]) -> float:
    """`gamma = somme des eta_a^2 / h_a` (13.3.3), en siemens par metre."""
    for _, frottement in especes:
        if frottement <= 0:
            msg = f"frottement nul ou negatif : h = {frottement}"
            raise ValueError(msg)
    return sum(charge**2 / frottement for charge, frottement in especes)


def champ_d_ohm(resistivite: Any, courant: Triplet) -> Triplet:  # noqa: ANN401 -- reel ou symbole
    """`E_s + E_e + E_d = rho i_cd` (13.3.5), en volts par metre."""
    x, y, z = (resistivite * composante for composante in courant)
    return (x, y, z)


def puissance_volumique_de_joule(resistivite: Any, courant: Triplet) -> Any:  # noqa: ANN401 -- reel ou symbole
    """`p_j = rho i_cd^2` (13.4.2), en watts par metre cube."""
    return resistivite * sum(composante**2 for composante in courant)


def _divergence(champ: Triplet) -> sp.Expr:
    paires = zip(champ, CARTESIENNES, strict=True)
    return sp.Add(*(sp.diff(sp.sympify(c), axe) for c, axe in paires))


def charge_volumique_apparente(
    resistivite: sp.Expr, conduction: Triplet, convection: Triplet, moteur: Triplet
) -> sp.Expr:
    """`eta + eta'` de (13.5.1), en C/m^3.

    `eps_0 [i_cd . grad(rho) - rho div(i_cv) - div(E_e + E_d)]` : la
    stationnarite, `div(i_cd + i_cv) = 0`, y est deja employee.
    """
    gradient = (sp.diff(resistivite, axe) for axe in CARTESIENNES)
    pente = sp.Add(*(i * g for i, g in zip(conduction, gradient, strict=True)))
    reste = resistivite * _divergence(convection) + _divergence(moteur)
    return sp.simplify(PERMITTIVITE * (pente - reste))


def champ_au_bord_du_dielectrique(
    resistivite: float,
    courant: float,
    *,
    charge_surfacique: float,
    permittivite: float,
) -> tuple[float, float]:
    """`(E_t, E_n) = (rho i_cd, sigma / epsilon)` (13.6.4), en V/m."""
    if permittivite <= 0:
        msg = f"permittivite nulle ou negative : {permittivite} F/m"
        raise ValueError(msg)
    return resistivite * courant, charge_surfacique / permittivite
