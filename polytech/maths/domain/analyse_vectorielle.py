"""Le gradient, la divergence, le rotationnel et le laplacien.

En coordonnees cartesiennes `(x, y, z)` :

- le **gradient** d'un champ scalaire, `(d(phi)/dx, d(phi)/dy, d(phi)/dz)`,
  generalise la derivee : il est perpendiculaire aux surfaces de niveau et
  pointe vers la plus forte pente ;
- la **divergence** d'un champ de vecteurs, `dF_x/dx + dF_y/dy + dF_z/dz`,
  dit s'il s'ecarte d'un point ou y converge. Son integrale sur un volume
  est le flux sortant du champ : c'est le theoreme de la divergence ;
- le **rotationnel** dit si les lignes du champ tournent autour d'un
  point ;
- le **laplacien** est la divergence du gradient.

Deux identites en decoulent : le rotationnel d'un gradient est nul, et la
divergence d'un rotationnel aussi.

Plusieurs cours emploient ces operateurs -- la mecanique des fluides,
l'electromagnetisme, la thermique. Ils vivent ici, dans leur discipline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence

    Vecteur = tuple[sp.Expr, sp.Expr, sp.Expr]

DIMENSION = 3


def _exiger_l_espace(coordonnees: Sequence[sp.Symbol]) -> None:
    if len(coordonnees) != DIMENSION:
        msg = f"trois coordonnees attendues, et non {len(coordonnees)}"
        raise ValueError(msg)


def gradient(scalaire: sp.Expr, coordonnees: Sequence[sp.Symbol]) -> Vecteur:
    """`grad(phi)`."""
    _exiger_l_espace(coordonnees)
    x, y, z = (sp.diff(sp.sympify(scalaire), axe) for axe in coordonnees)
    return (x, y, z)


def divergence(champ: Sequence[sp.Expr], coordonnees: Sequence[sp.Symbol]) -> sp.Expr:
    """`div F`."""
    _exiger_l_espace(coordonnees)
    if len(champ) != DIMENSION:
        msg = f"autant de composantes que de coordonnees attendues : {len(champ)}"
        raise ValueError(msg)
    paires = zip(champ, coordonnees, strict=True)
    return sp.Add(*(sp.diff(sp.sympify(c), axe) for c, axe in paires))


def rotationnel(champ: Sequence[sp.Expr], coordonnees: Sequence[sp.Symbol]) -> Vecteur:
    """`rot F`."""
    _exiger_l_espace(coordonnees)
    f_x, f_y, f_z = (sp.sympify(composante) for composante in champ)
    x, y, z = coordonnees
    return (
        sp.diff(f_z, y) - sp.diff(f_y, z),
        sp.diff(f_x, z) - sp.diff(f_z, x),
        sp.diff(f_y, x) - sp.diff(f_x, y),
    )


def laplacien(scalaire: sp.Expr, coordonnees: Sequence[sp.Symbol]) -> sp.Expr:
    """`div(grad(phi))`."""
    return divergence(gradient(scalaire, coordonnees), coordonnees)
