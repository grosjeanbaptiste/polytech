"""Lire une equation aux derivees partielles : ordre, linearite, solutions.

Mathematiques appliquees, 9.1 :

* l'**ordre** est « le plus haut degre de derivation (partiel) qui y
  apparait » ;
* l'equation est **lineaire** si elle est une somme de termes ou
  n'intervient qu'une seule derivee partielle de `u`, ou `u` elle-meme,
  a la puissance 1 ;
* elle est **homogene** s'il n'y a « pas de terme ne contenant ni `u` ni
  aucune de ses derivees ».

Sa solution generale contient des fonctions arbitraires : la verifier,
c'est la substituer.

A coefficients variables, le type d'une equation du second ordre (9.4)
change avec le point : il suit le signe de `ab - h^2` en ce point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.edp.classification import (
    Type,
    discriminant,
    type_a_deux_variables,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def caracteriser_l_edp(equation: sp.Expr, inconnue: sp.Expr) -> tuple[int, bool, bool]:
    """`(ordre, lineaire, homogene)` de l'equation `equation = 0`."""
    derivees = sorted(equation.atoms(sp.Derivative), key=sp.default_sort_key)
    if not derivees:
        msg = f"aucune derivee partielle dans {equation} : ce n'est pas une EDP"
        raise ValueError(msg)
    ordre = max(derivee.derivative_count for derivee in derivees)
    termes = [*derivees, inconnue]
    muets = sp.symbols(f"v0:{len(termes)}")
    polynome = sp.expand(equation.subs(dict(zip(termes, muets, strict=True))))
    lineaire = bool(polynome.is_polynomial(*muets)) and (
        sp.Poly(polynome, *muets).total_degree() <= 1
    )
    libre = polynome.subs(dict.fromkeys(muets, 0))
    return ordre, lineaire, bool(sp.simplify(libre) == 0)


def est_solution_de_l_edp(
    equation: sp.Expr, inconnue: sp.Expr, candidate: sp.Expr
) -> bool:
    """Vrai si `candidate`, mise a la place de l'inconnue, annule l'equation."""
    return bool(sp.simplify(equation.subs(inconnue, candidate)) == 0)


def types_selon_le_point(a: object, h: object, b: object) -> dict[Type, sp.Basic]:
    """La condition sur le point qui fait chacun des trois types."""
    valeur = discriminant(a, h, b)
    return {
        Type.ELLIPTIQUE: valeur > 0,
        Type.PARABOLIQUE: sp.Eq(valeur, 0),
        Type.HYPERBOLIQUE: valeur < 0,
    }


def type_au_point(
    a: object, h: object, b: object, point: Mapping[sp.Symbol, object]
) -> Type:
    """Le type en un point donne ; leve si le point ne fixe pas le signe."""
    coefficients = (sp.sympify(c).subs(point) for c in (a, h, b))
    return type_a_deux_variables(*coefficients)
