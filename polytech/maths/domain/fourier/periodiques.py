"""Les fonctions periodiques.

`f` est periodique de periode `T > 0` si `f(x + T) = f(x)` pour tout `x`.
`e^(lambda x)` ne l'est que pour `lambda` imaginaire pur, `lambda = j w` :
sa periode est `2 pi / |w|`, et `w` est la pulsation.

L'integrale d'une fonction periodique sur une periode ne depend pas du
point de depart.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE


def est_periodique(fonction: sp.Expr, periode: object) -> bool:
    """Vrai si `f(x + T) = f(x)`."""
    decalee = fonction.subs(VARIABLE, VARIABLE + sp.sympify(periode))
    return bool(sp.simplify(decalee - fonction) == 0)


def periode_de_l_exponentielle(exposant: object) -> sp.Expr:
    """`2 pi / |w|` pour `e^(j w x)` ; leve si l'exposant n'est pas imaginaire pur."""
    valeur = sp.sympify(exposant)
    pulsation = sp.im(valeur)
    if sp.re(valeur) != 0 or pulsation == 0:
        msg = (
            f"exposant {valeur} pas imaginaire pur : e^(lambda x) n'est pas periodique"
        )
        raise ValueError(msg)
    return 2 * sp.pi / sp.Abs(pulsation)


def integrale_sur_une_periode(
    fonction: sp.Expr, depart: object, periode: object
) -> sp.Expr:
    """`int_x^(x+T) f`, qui ne depend pas de `x`."""
    debut = sp.sympify(depart)
    return sp.simplify(sp.integrate(fonction, (VARIABLE, debut, debut + periode)))
