"""Le residu lu sur la serie de Laurent : le coefficient d'indice -1.

Le residu de `f` en `a` est le coefficient `a_(-1)` de sa serie de Laurent
centree en `a` et de rayon interieur nul. Deux facons de l'ecrire :

- la serie n'a qu'un nombre fini de termes singuliers (un pole) : un
  developpement fini en `z - a` suffit ;
- elle en a une infinite (singularite essentielle) : on pose `w = 1 / (z - a)`
  et l'on developpe en `w`. Cela ne vaut que si `a` est la seule
  singularite de `f` a distance finie -- sinon le developpement en `w`
  decrit une autre couronne, et son coefficient n'est pas le residu.

Source : Mathematiques pour l'ingenieur III, syllabus, section 9.2,
methode 5.
"""

from __future__ import annotations

import sympy as sp
from sympy.calculus.singularities import singularities

from polytech.maths.domain.analyse_complexe.singularites import (
    ECART,
    autour_de,
    ordre_du_pole,
)

INVERSE = sp.Symbol("w")
"""`1 / (z - a)`."""


def _par_l_inverse(fonction: sp.Expr, variable: sp.Symbol, point: object) -> sp.Expr:
    """Le coefficient de `w` dans le developpement en `w = 1 / (z - a)`."""
    en_w = sp.simplify(autour_de(fonction, variable, point).subs(ECART, 1 / INVERSE))
    ailleurs = singularities(en_w, INVERSE, sp.S.Complexes) - sp.FiniteSet(0)
    if ailleurs != sp.EmptySet:
        msg = (
            f"serie de Laurent de {fonction} en {point} hors de portee : "
            f"infinite de termes singuliers, et d'autres singularites ({ailleurs})"
        )
        raise ValueError(msg)
    return sp.series(en_w, INVERSE, 0, 2).removeO().coeff(INVERSE, 1)


def residu_par_laurent(
    fonction: sp.Expr, variable: sp.Symbol, point: object
) -> sp.Expr:
    """Le coefficient de `1 / (z - a)` dans la serie de Laurent de `f` en `a`."""
    try:
        ordre_du_pole(fonction, variable, point)
    except ValueError:
        return _par_l_inverse(fonction, variable, point)
    locale = autour_de(fonction, variable, point)
    return sp.series(locale, ECART, 0, 1).removeO().coeff(ECART, -1)
