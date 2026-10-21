"""La division des polynomes et l'ordre d'une racine, en ecriture exacte.

Le syllabus d'algebre (IV.1) divise de trois facons : le calcul ecrit,
pour un diviseur quelconque ; le tableau de Horner, quand le diviseur est
`x - a` ; les deductions successives. Les trois rendent le meme quotient
et le meme reste, `P = Q D + R` avec `deg R < deg D`.

Une racine `a` est d'ordre `k` quand `(x - a)^k` divise `P` et que
`(x - a)^(k+1)` ne le divise pas : `P`, `P'`, ..., `P^(k-1)` s'annulent en
`a`, et `P^(k)` non.

`polynomes.py` porte la division en flottants ; ici les coefficients
restent des fractions, comme dans le calcul ecrit de la page 48.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE

if TYPE_CHECKING:
    from collections.abc import Sequence


def division(
    dividende: sp.Expr, diviseur: sp.Expr, variable: sp.Symbol = VARIABLE
) -> tuple[sp.Expr, sp.Expr]:
    """Le quotient et le reste de la division euclidienne."""
    if sp.simplify(diviseur) == 0:
        msg = "diviseur nul : la division n'existe pas"
        raise ValueError(msg)
    quotient, reste = sp.div(sp.expand(dividende), sp.expand(diviseur), variable)
    return sp.expand(quotient), sp.expand(reste)


def tableau_de_horner(
    coefficients: Sequence[sp.Expr], racine: sp.Expr
) -> tuple[tuple[sp.Expr, ...], tuple[sp.Expr, ...]]:
    """Les deux lignes du tableau : les produits, puis les sommes.

    La derniere somme est le reste ; les autres sont les coefficients du
    quotient par `x - racine`.
    """
    produits: list[sp.Expr] = []
    sommes = [sp.sympify(coefficients[0])]
    for coefficient in coefficients[1:]:
        produits.append(sommes[-1] * racine)
        sommes.append(coefficient + produits[-1])
    return tuple(produits), tuple(sommes)


def ordre_de_la_racine(
    polynome: sp.Expr, racine: sp.Expr, variable: sp.Symbol = VARIABLE
) -> int:
    """Le nombre de derivees successives, `P` comprise, qui s'annulent en `racine`."""
    courant = sp.expand(polynome)
    if courant == 0:
        msg = "polynome nul : toute valeur en est racine, a tout ordre"
        raise ValueError(msg)
    ordre = 0
    while sp.simplify(courant.subs(variable, racine)) == 0:
        ordre += 1
        courant = sp.diff(courant, variable)
    return ordre


def par_deductions_successives(
    dividende: sp.Expr, diviseur: sp.Expr, variable: sp.Symbol = VARIABLE
) -> tuple[tuple[sp.Expr, ...], sp.Expr]:
    """Les coefficients du quotient, deduits un a un du plus haut, et le reste.

    Chaque coefficient est choisi pour que le terme de meme degre du produit
    `diviseur . quotient` soit celui du dividende (IV.1, page 49).
    """
    d = sp.Poly(diviseur, variable)
    if d.degree() < 1:
        msg = f"le diviseur doit etre de degre au moins 1 : {diviseur}"
        raise ValueError(msg)
    reste = sp.Poly(dividende, variable)
    coefficients: list[sp.Expr] = []
    for degre in range(reste.degree() - d.degree(), -1, -1):
        coefficient = reste.coeff_monomial(variable ** (degre + d.degree())) / d.LC()
        coefficients.append(sp.nsimplify(coefficient))
        reste -= sp.Poly(coefficient * variable**degre, variable) * d
    return tuple(coefficients), sp.expand(reste.as_expr())
