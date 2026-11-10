"""La distribution associee a une fonction, et les fonctions a support compact.

`D(R)` regroupe les fonctions lisses a **support compact** : nulles hors d'un
intervalle `[a, b]` de longueur finie. Une fonction `f` y definit la
fonctionnelle `<T_f, phi> = int f phi dx`, dont l'integrale se limite au
support de `phi` (syllabus, 10.7.1 a 10.7.3). `exp(-x^2)` est de decroissance
rapide, donc dans `S(R)`, mais n'est pas a support compact.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE


def _ensemble_ou_non_nulle(fonction: sp.Expr) -> sp.Set | None:
    """Les `x` ou une branche non nulle s'applique ; `None` si elle n'a pas de forme."""
    if fonction == 0:
        return sp.S.EmptySet
    if not isinstance(fonction, sp.Piecewise):
        return None
    morceaux = (
        sp.solveset(condition, VARIABLE, sp.S.Reals)
        for expression, condition in fonction.args
        if expression != 0
    )
    return sp.Union(*morceaux)


def support_compact(fonction: sp.Expr) -> sp.Set | None:
    """Un ensemble borne hors duquel `phi` est nulle, ou `None` s'il n'y en a pas."""
    ensemble = _ensemble_ou_non_nulle(sp.piecewise_fold(sp.sympify(fonction)))
    if ensemble is None:
        return None
    if ensemble.is_empty:
        return sp.S.EmptySet
    borne = ensemble.inf.is_finite and ensemble.sup.is_finite
    return ensemble if borne else None


def appliquer_la_distribution_reguliere(fonction: sp.Expr, test: sp.Expr) -> sp.Expr:
    """`<T_f, phi> = int f phi dx`, sur le support de `phi` ; leve sans support."""
    support = support_compact(test)
    if support is None:
        msg = f"{test} n'est pas a support compact : pas une fonction test de D(R)"
        raise ValueError(msg)
    if support.is_empty:
        return sp.Integer(0)
    produit = sp.piecewise_fold(sp.sympify(fonction) * test)
    return sp.simplify(sp.integrate(produit, (VARIABLE, support.inf, support.sup)))
