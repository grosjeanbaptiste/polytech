"""La polarisation, les charges fictives et le deplacement electrique.

Le syllabus d'electricite generale (2.8, 2.9 et 2.13) :

* la polarisation est le moment dipolaire volumique, `P = somme(p_i) / Delta V`
  (2.8.9) ; elle est nulle si les dipoles sont orientes au hasard ;
* son potentiel est celui de charges **fictives**, `eta' = -div P` en volume
  (2.9.1a) et `sigma' = -u_n . P` en surface, `u_n` etant la normale
  orientee vers l'interieur du milieu (2.9.1b). Leur total est nul (2.9.4) ;
* le deplacement electrique `D = eps_0 E + P` (2.13.3) ne voit que les
  charges reelles : `div D = eta` (2.13.5), alors que
  `eps_0 div E = eta + eta'` (2.15.1). `E_i = P / eps_0` est l'electrisation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.electrostatique.lois import (
    divergence_cartesienne,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.electromagnetisme.domain.electrostatique.moments import Point
    from polytech.electromagnetisme.domain.electrostatique.potentiel import Vecteur


def _vecteur(composantes: Point) -> Vecteur:
    x, y, z = (sp.sympify(c) for c in composantes)
    return (x, y, z)


def polarisation(moments: Sequence[Point], volume: object) -> Vecteur:
    """`P = somme(p_i) / Delta V`."""
    etendue = sp.sympify(volume)
    if etendue.is_positive is not True:
        msg = f"volume strictement positif attendu, et non {etendue}"
        raise ValueError(msg)
    x, y, z = (
        sum((sp.sympify(p[axe]) for p in moments), sp.Integer(0)) / etendue
        for axe in range(3)
    )
    return (x, y, z)


def charge_volumique_fictive(polarise: Point) -> sp.Expr:
    """`eta' = -div P`."""
    return -divergence_cartesienne(_vecteur(polarise))


def charge_surfacique_fictive(polarise: Point, *, normale_interieure: Point) -> sp.Expr:
    """`sigma' = -u_n . P`, `u_n` pointant vers l'interieur du milieu polarise."""
    normale = _vecteur(normale_interieure)
    if sp.simplify(sum(c**2 for c in normale) - 1) != 0:
        msg = f"normale unitaire attendue, et non {normale_interieure}"
        raise ValueError(msg)
    return -sum(n * p for n, p in zip(normale, _vecteur(polarise), strict=True))


def deplacement(champ: Point, polarise: Point) -> Vecteur:
    """`D = eps_0 E + P`."""
    x, y, z = (
        PERMITTIVITE * e + p
        for e, p in zip(_vecteur(champ), _vecteur(polarise), strict=True)
    )
    return (x, y, z)


def electrisation(polarise: Point) -> Vecteur:
    """`E_i = P / eps_0`."""
    x, y, z = (p / PERMITTIVITE for p in _vecteur(polarise))
    return (x, y, z)
