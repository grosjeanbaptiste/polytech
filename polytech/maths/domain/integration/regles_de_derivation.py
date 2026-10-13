"""Les sept regles de derivation du formulaire.

Section 3.1 de `Formulaire_Aide_memoire.pdf` (page 4) :

    (f +- g)' = f' +- g'            (f o g)' = (f' o g) . g'
    (alpha f)' = alpha f'           (f^-1)' = 1 / (f' o f^-1)
    (f g)' = f' g + f g'            (f^g)' = f^g (g' ln f + g f' / f)
    (f / g)' = (f' g - f g') / g^2

Chaque fonction applique **sa** regle aux derivees des morceaux : elle ne
derive jamais l'ensemble. C'est ce qui permet a un test de la comparer a
la derivee de l'ensemble, calculee autrement.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE as X


def _prime(fonction: object) -> sp.Expr:
    return sp.diff(sp.sympify(fonction), X)


def de_la_somme(f: object, g: object) -> sp.Expr:
    """`(f + g)' = f' + g'`."""
    return _prime(f) + _prime(g)


def du_multiple(alpha: object, f: object) -> sp.Expr:
    """`(alpha f)' = alpha f'`, pour un reel `alpha`."""
    return sp.sympify(alpha) * _prime(f)


def du_produit(f: object, g: object) -> sp.Expr:
    """`(f g)' = f' g + f g'`."""
    return _prime(f) * sp.sympify(g) + sp.sympify(f) * _prime(g)


def du_quotient(f: object, g: object) -> sp.Expr:
    """`(f / g)' = (f' g - f g') / g^2`."""
    diviseur = sp.sympify(g)
    if diviseur == 0:
        msg = "denominateur nul : le quotient n'existe pas"
        raise ValueError(msg)
    return (_prime(f) * diviseur - sp.sympify(f) * _prime(g)) / diviseur**2


def de_la_composee(f: object, g: object) -> sp.Expr:
    """`(f o g)' = (f' o g) . g'`, `f` et `g` etant ecrites en `x`."""
    return _prime(f).subs(X, sp.sympify(g)) * _prime(g)


def de_la_reciproque(f: object, reciproque: object) -> sp.Expr:
    """`(f^-1)' = 1 / (f' o f^-1)` ; leve si `f o f^-1` n'est pas `x`."""
    inverse = sp.sympify(reciproque)
    if sp.simplify(sp.sympify(f).subs(X, inverse) - X) != 0:
        msg = f"{inverse} n'est pas la reciproque de {f} : f o f^-1 ne rend pas x"
        raise ValueError(msg)
    return 1 / _prime(f).subs(X, inverse)


def de_la_puissance(f: object, g: object) -> sp.Expr:
    """`(f^g)' = f^g (g' ln f + g f' / f)`."""
    base, exposant = sp.sympify(f), sp.sympify(g)
    return base**exposant * (_prime(g) * sp.log(base) + exposant * _prime(f) / base)
