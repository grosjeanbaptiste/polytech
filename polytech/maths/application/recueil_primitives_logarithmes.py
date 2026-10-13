"""Ecrire les logarithmes d'une primitive : valeur absolue, argument sans facteur.

La primitive de `1/u` est `ln |u|`, des deux cotes de zero. Et l'integration
terme a terme d'une fraction laisse `ln |8x - 16|` la ou l'on attend
`ln |x - 2|` : les deux ne different que d'une constante, que `K` absorbe.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE as X


def _unitaire(argument: sp.Expr) -> sp.Expr:
    """L'argument d'un logarithme, sans facteur constant : `8x - 16` devient `x - 2`."""
    if not (argument.is_polynomial(X) and argument.has(X)):
        return argument
    _, primitif = sp.Poly(argument, X).primitive()
    return primitif.as_expr()


def _terme_sous_valeur_absolue(terme: sp.Expr) -> sp.Expr:
    coefficient, reste = terme.as_independent(X)
    if isinstance(reste, sp.log):
        return coefficient * sp.log(sp.Abs(_unitaire(reste.args[0])))
    return terme.replace(sp.log, lambda argument: sp.log(sp.Abs(argument)))


def valeur_absolue(expression: sp.Expr) -> sp.Expr:
    """`ln u` devient `ln |u|`, sans le facteur constant que l'integration y laisse.

    La primitive de `1/u` vaut des deux cotes de zero. Et `c ln |8x - 16|` ne
    differe de `c ln |x - 2|` que d'une constante, que `K` absorbe -- tant que
    `c` est une constante : sous un facteur qui depend de `x`, l'argument reste.
    """
    termes = sp.Add.make_args(expression)
    return sp.Add(*[_terme_sous_valeur_absolue(terme) for terme in termes])
