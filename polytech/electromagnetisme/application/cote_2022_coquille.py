"""Exercice cote 2022 (Charleroi), question 1 : une coquille spherique chargee.

Theoreme de Gauss sur une sphere de rayon `r` : le flux de `D = eps E`
vaut la charge enfermee. La symetrie spherique rend `E` radial et de
module constant sur la sphere ; le flux est alors `eps E 4 pi r^2`.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.images import PERMITTIVITE

EPS0: Final = PERMITTIVITE
EPS: Final = sp.Symbol("varepsilon", positive=True)
"""La permittivite de la coquille."""
ETA, Q = sp.symbols("eta q", positive=True)
RAYON, R1, R2 = sp.symbols("r R_1 R_2", positive=True)


def _volume(interieur: sp.Expr, exterieur: sp.Expr) -> sp.Expr:
    """Le volume compris entre deux spheres concentriques."""
    return sp.Rational(4, 3) * sp.pi * (exterieur**3 - interieur**3)


def _champ(charge_enfermee: sp.Expr, permittivite: sp.Expr) -> sp.Expr:
    """`E = Q_int / (eps 4 pi r^2)`."""
    return sp.simplify(charge_enfermee / (permittivite * 4 * sp.pi * RAYON**2))


def champs() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """`E(r)` dans la cavite, dans la coquille, puis au dehors."""
    cavite = _champ(sp.Integer(0), EPS0)
    coquille = _champ(ETA * _volume(R1, RAYON), EPS)
    dehors = _champ(ETA * _volume(R1, R2), EPS0)
    return cavite, coquille, dehors


def _divergence(deplacement_radial: sp.Expr) -> sp.Expr:
    """`(1/r^2) d(r^2 D_r)/dr`, la divergence d'un champ radial."""
    return sp.simplify(sp.diff(RAYON**2 * deplacement_radial, RAYON) / RAYON**2)


def divergences() -> tuple[sp.Expr, sp.Expr]:
    """`div D` dans la coquille, puis au dehors : la charge volumique locale."""
    _, coquille, dehors = champs()
    return _divergence(EPS * coquille), _divergence(EPS0 * dehors)


def charges_surfaciques() -> tuple[sp.Expr, sp.Expr]:
    """Coquille conductrice neutre, `q` au centre : faces interieure, puis exterieure.

    Une sphere de Gauss prise dans le metal, ou le champ est nul, n'enferme
    aucune charge nette : la face interieure porte `-q`. La coquille etant
    neutre, la face exterieure porte `+q`.
    """
    interieure = sp.solve(
        sp.Eq(Q + sp.Symbol("s") * 4 * sp.pi * R1**2, 0), sp.Symbol("s")
    )[0]
    exterieure = -interieure * R1**2 / R2**2
    return interieure, sp.simplify(exterieure)
