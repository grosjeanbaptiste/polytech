"""Le type d'une equation lineaire du second ordre, et ce qu'il commande.

`a u_xx + 2h u_xy + b u_yy + ... = 0` se classe comme la conique
`a x^2 + 2h x y + b y^2 + ... = 0`, sur le signe de `ab - h^2` :
**elliptique** s'il est positif (Laplace), **parabolique** s'il est nul
(chaleur), **hyperbolique** s'il est negatif (ondes). A plus de deux
variables, ce sont les signes des valeurs propres de la forme quadratique
des derivees secondes qui decident.

Le type fixe la forme des solutions et les conditions aux frontieres qui
font un probleme bien pose : Dirichlet ou Neumann pour une equation
elliptique, Cauchy pour une hyperbolique.
"""

from __future__ import annotations

from enum import Enum

import sympy as sp


class Type(Enum):
    """Les trois types, nommes d'apres les coniques."""

    ELLIPTIQUE = "elliptique"
    PARABOLIQUE = "parabolique"
    HYPERBOLIQUE = "hyperbolique"


class Condition(Enum):
    """Les conditions aux frontieres les plus courantes."""

    DIRICHLET = "la valeur de u sur le bord"
    NEUMANN = "la derivee normale de u sur le bord"
    CAUCHY = "u et sa derivee en temps a l'instant initial"


def discriminant(a: object, h: object, b: object) -> sp.Expr:
    """`ab - h^2`."""
    return sp.expand(sp.sympify(a) * sp.sympify(b) - sp.sympify(h) ** 2)


def type_a_deux_variables(a: object, h: object, b: object) -> Type:
    """Le type selon le signe de `ab - h^2` ; leve si ce signe depend du point."""
    valeur = sp.simplify(discriminant(a, h, b))
    if valeur == 0:
        return Type.PARABOLIQUE
    if valeur.is_positive:
        return Type.ELLIPTIQUE
    if valeur.is_negative:
        return Type.HYPERBOLIQUE
    msg = f"signe du discriminant indetermine : {valeur} change avec le point"
    raise ValueError(msg)


def type_par_la_forme_quadratique(matrice: sp.Matrix) -> Type:
    """Le type selon les signes des valeurs propres de la forme quadratique."""
    valeurs = list(matrice.eigenvals())
    if any(valeur == 0 for valeur in valeurs):
        return Type.PARABOLIQUE
    positives = [bool(valeur.is_positive) for valeur in valeurs]
    return Type.ELLIPTIQUE if len(set(positives)) == 1 else Type.HYPERBOLIQUE


def conditions_adaptees(nature: Type) -> tuple[Condition, ...]:
    """Les conditions qui assurent l'unicite et la stabilite de la solution."""
    if nature is Type.ELLIPTIQUE:
        return (Condition.DIRICHLET, Condition.NEUMANN)
    if nature is Type.HYPERBOLIQUE:
        return (Condition.CAUCHY,)
    msg = "conditions du type parabolique : le syllabus ne le dit pas"
    raise ValueError(msg)
