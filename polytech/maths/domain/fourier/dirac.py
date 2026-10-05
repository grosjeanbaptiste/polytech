"""La distribution de Dirac : sa derivee, et sa composee avec une fonction.

`<delta_a, phi> = phi(a)` vit dans `distributions`. Deux regles s'y
ajoutent, que le cours d'electricite generale rappelle a son paragraphe 1.6
pour ecrire des charges ponctuelles :

* la derivee, par transposition : `<delta_a', phi> = -phi'(a)` ;
* la composee : `delta(f(x)) = somme des delta(x - x_i) / |f'(x_i)|`, la
  somme portant sur les zeros `x_i` de `f` -- pourvu qu'ils soient simples,
  sans quoi `f'(x_i)` s'annule.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE


def appliquer_la_derivee_du_dirac(point: object, fonction: sp.Expr) -> sp.Expr:
    """`<delta_a', phi> = -phi'(a)`."""
    derivee = sp.diff(sp.sympify(fonction), VARIABLE)
    return -derivee.subs(VARIABLE, sp.sympify(point))


def dirac_d_une_fonction(fonction: sp.Expr) -> dict[sp.Expr, sp.Expr]:
    """Les pics de `delta(f(x))` : chaque zero reel de `f`, et son poids."""
    expression = sp.sympify(fonction)
    derivee = sp.diff(expression, VARIABLE)
    pics: dict[sp.Expr, sp.Expr] = {}
    for zero in sp.solveset(expression, VARIABLE, domain=sp.S.Reals):
        pente = derivee.subs(VARIABLE, zero)
        if pente == 0:
            msg = f"zero multiple en {zero} : la derivee s'y annule"
            raise ValueError(msg)
        pics[zero] = 1 / sp.Abs(pente)
    return pics
