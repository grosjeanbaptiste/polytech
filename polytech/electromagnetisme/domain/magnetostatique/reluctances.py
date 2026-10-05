"""La reluctance d'un tube de flux, et les associations de reluctances.

Le syllabus d'electricite generale (10.11 a 10.14). Le long d'un tube de
flux, dans un milieu sans courant, la difference de potentiel magnetique
vaut `U_12 = R_12 Phi` (10.11.11) : la loi d'Hopkinson. La reluctance du
troncon est `R_12 = integrale de dl / (mu S)` (10.11.10), soit
`l / (mu S)` si le champ est uniforme (10.11.12) ; son inverse est la
permeance.

Pour un tube ferme, `U` est la circulation du champ, donc le courant
enlace : `F = somme des N I = R Phi` (10.11.15), la force magnetomotrice,
en amperes-tours. Les reluctances traversees par un meme flux s'ajoutent
(10.12.1) ; celles que soumet une meme difference de potentiel ajoutent
leurs inverses (10.12.2).

Un circuit reel perd des lignes d'induction : `N I = R_u Phi_u = R_d
Phi_d` (10.14.1), le tube de dispersion passant par l'air.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence


def reluctance_d_un_tube(
    section: sp.Expr, abscisse: sp.Symbol, longueur: object, permeabilite: object
) -> sp.Expr:
    """`integrale de 0 a l de ds / (mu S(s))` (10.11.10), en A/Wb."""
    if permeabilite == 0:
        msg = "permeabilite nulle : le tube n'aurait aucun flux a conduire"
        raise ValueError(msg)
    integrande = 1 / (sp.sympify(permeabilite) * section)
    return sp.simplify(sp.integrate(integrande, (abscisse, 0, longueur)))


def _positives(reluctances: Sequence[float]) -> Sequence[float]:
    if not reluctances:
        msg = "aucune reluctance a associer"
        raise ValueError(msg)
    for valeur in reluctances:
        if valeur <= 0:
            msg = f"reluctance nulle ou negative : {valeur} A/Wb"
            raise ValueError(msg)
    return reluctances


def reluctance_en_serie(*reluctances: float) -> float:
    """`R = somme des R_k` (10.12.1)."""
    return sum(_positives(reluctances))


def reluctance_en_parallele(*reluctances: float) -> float:
    """`1 / R = somme des 1 / R_k` (10.12.2)."""
    return 1 / sum(1 / valeur for valeur in _positives(reluctances))


def permeance(reluctance: float) -> float:
    """`Lambda = 1 / R`, en webers par ampere."""
    return 1 / _positives((reluctance,))[0]


def force_magnetomotrice(bobines: Sequence[tuple[int, float]]) -> float:
    """`F = somme des N I` (10.11.15) : chaque courant compte avec son sens."""
    return sum(spires * courant for spires, courant in bobines)


def flux_utile_et_de_dispersion(
    amperes_tours: float, utile: float, dispersion: float
) -> tuple[float, float]:
    """`(Phi_u, Phi_d)` de (10.14.1) : la meme force sur deux tubes."""
    _positives((utile, dispersion))
    return amperes_tours / utile, amperes_tours / dispersion
