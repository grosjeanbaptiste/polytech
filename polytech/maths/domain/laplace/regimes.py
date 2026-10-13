"""Le regime et le transitoire d'une reponse ; l'indicielle et l'impulsionnelle.

Mathematiques appliquees, 3.6 : la reponse d'un systeme stable a une
sollicitation se lit en deux parts. Le **transitoire** s'eteint -- chacun
de ses termes tend vers 0 -- et le **regime** est ce qui demeure : une
constante (« en regime, `lim y(t) = 1/2` »), une sinusoide.

Un systeme de **transmittance** `G(p)` repond a l'impulsion par
`L^-1(G)` et a l'echelon par `L^-1(G / p)`. La seconde est la primitive
de la premiere : « la derivee de la reponse indicielle est la reponse
impulsionnelle » (exemple 3.2).
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T


def _termes(reponse: sp.Expr) -> tuple[sp.Expr, ...]:
    return tuple(sp.Add.make_args(sp.expand(reponse)))


def _s_eteint(terme: sp.Expr) -> bool:
    """Vrai si le terme tend vers 0 ; leve s'il n'est pas borne."""
    limite = sp.limit(terme, T, sp.oo)
    if limite.has(sp.oo, -sp.oo, sp.zoo):
        msg = f"le terme {terme} diverge : la reponse n'a pas de regime"
        raise ValueError(msg)
    return bool(limite == 0)


def regime_transitoire(reponse: sp.Expr) -> sp.Expr:
    """La somme des termes qui tendent vers 0."""
    eteints = (terme for terme in _termes(reponse) if _s_eteint(terme))
    return sum(eteints, sp.Integer(0))


def regime_permanent(reponse: sp.Expr) -> sp.Expr:
    """Ce qui reste quand le transitoire s'est eteint."""
    return sp.expand(reponse) - regime_transitoire(reponse)


def reponse_impulsionnelle(transmittance: sp.Expr) -> sp.Expr:
    """`L^-1(G)` : la reponse a l'impulsion de Dirac, pour `t > 0`."""
    originale = sp.inverse_laplace_transform(transmittance, P, T, noconds=True)
    return sp.simplify(originale.subs(sp.Heaviside(T), 1))


def reponse_indicielle(transmittance: sp.Expr) -> sp.Expr:
    """`L^-1(G / p)` : la reponse a l'echelon."""
    return reponse_impulsionnelle(transmittance / P)
