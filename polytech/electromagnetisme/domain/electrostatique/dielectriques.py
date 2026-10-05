"""Les dielectriques : permittivite, charges fictives, energie, hysteresis.

Le syllabus d'electricite generale (chapitre 4), pour un dielectrique
homogene, isotrope et lineaire :

* `D = eps_0 E + P = eps E` (4.2.1, 4.2.2), avec `eps = eps_r eps_0` et la
  susceptibilite `chi = eps_r - 1` telle que `P = chi eps_0 E` (4.2.4,
  4.2.5). `P` suit `E`, donc `eps_r >= 1` ;
* `P = (1 - eps_0 / eps) D` (4.4.1) : les charges fictives sont
  proportionnelles aux charges reelles, `eta' = -(1 - eps_0 / eps) eta`
  (4.4.2), et la charge apparente vaut `eta / eps_r` (4.4.3). Les memes
  relations valent pour les charges surfaciques (4.4.4, 4.4.5) ;
* le travail volumique qu'il faut fournir vaut `dw = E . dD` (4.3.6), soit
  `w = integrale de E dD` (4.3.8) -- `E . D / 2` si le milieu est lineaire
  (4.3.11). Sur un cycle d'hysteresis, ce travail est l'aire du cycle : il
  est dissipe en chaleur (4.8).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.symboles import TEMPS

if TYPE_CHECKING:
    from polytech.electromagnetisme.domain.electrostatique.moments import Point
    from polytech.electromagnetisme.domain.electrostatique.potentiel import Vecteur

CHAMP_DISRUPTIF_DE_L_AIR: Final = 3e6
"""« Quelque 3 MV/m », aux conditions normales : au-dela, l'air s'ionise."""

PERMITTIVITE_RELATIVE_DE_L_AIR: Final = 1.0006
"""« Environ 1,000 6 » : l'air se comporte pratiquement comme le vide."""


def susceptibilite(relative: object) -> sp.Expr:
    """`chi = eps_r - 1` ; leve si la permittivite relative est inferieure a 1."""
    eps_r = sp.sympify(relative)
    if (eps_r - 1).is_negative:
        msg = f"permittivite relative inferieure a 1 : {eps_r}"
        raise ValueError(msg)
    return eps_r - 1


def polarisation_depuis_le_champ(relative: object, champ: Point) -> Vecteur:
    """`P = chi eps_0 E`."""
    facteur = susceptibilite(relative) * PERMITTIVITE
    x, y, z = (facteur * sp.sympify(composante) for composante in champ)
    return (x, y, z)


def polarisation_depuis_le_deplacement(relative: object, induction: object) -> sp.Expr:
    """`P = (1 - eps_0 / eps) D`."""
    eps_r = sp.sympify(relative)
    return susceptibilite(eps_r) / eps_r * sp.sympify(induction)


def charge_fictive(relative: object, reelle: object) -> sp.Expr:
    """`eta' = -(1 - eps_0 / eps) eta` ; la meme relation lie `sigma'` a `sigma`."""
    return -polarisation_depuis_le_deplacement(relative, reelle)


def charge_apparente(relative: object, reelle: object) -> sp.Expr:
    """`eta + eta' = eta / eps_r` : ce que le champ voit de la charge reelle."""
    return sp.sympify(reelle) + charge_fictive(relative, reelle)


def energie_volumique(champ: sp.Expr, induction: sp.Symbol) -> sp.Expr:
    """`w = integrale de 0 a D de E dD`, `E` etant donne en fonction de `D`."""
    muette = sp.Symbol("D_muette", positive=True)
    return sp.integrate(
        sp.sympify(champ).subs(induction, muette), (muette, 0, induction)
    )


def travail_volumique_du_cycle(
    champ: sp.Expr, induction: sp.Expr, periode: object
) -> sp.Expr:
    """`integrale de E dD` sur un cycle parcouru en une periode : son aire."""
    integrande = sp.sympify(champ) * sp.diff(sp.sympify(induction), TEMPS)
    return sp.simplify(sp.integrate(integrande, (TEMPS, 0, periode)))
