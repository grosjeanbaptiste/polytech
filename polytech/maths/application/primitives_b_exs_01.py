"""Mathematique, exercices 1, section 1.1 : les primitives 8 a 14."""

from __future__ import annotations

import sympy as sp

from polytech.maths.application.primitives_a_exs_01 import ANGLE, substitution
from polytech.maths.application.resolution_exs_01 import A, B, Resolution, par_equation
from polytech.maths.domain.integration.irrationnelles import par_changement_inverse
from polytech.maths.domain.integration.substitution import NOUVELLE as U
from polytech.maths.domain.integration.trigonometrie import (
    T_DEMI,
    par_la_tangente_de_l_arc_moitie,
)
from polytech.maths.domain.symboles import VARIABLE as X

_T = sp.Symbol("t", positive=True)
CHANGEMENT = "changement de variable"


def exercice_8() -> Resolution:
    """`e^(2x) / racine(e^x + 1)` : `u = e^x`, et il reste `u / racine(u + 1)`."""
    integrande = sp.exp(2 * X) / sp.sqrt(sp.exp(X) + 1)
    reecrit = sp.exp(X) * sp.exp(X) / sp.sqrt(sp.exp(X) + 1)
    return substitution(integrande, reecrit, sp.exp(X))


def exercice_9() -> tuple[Resolution, Resolution]:
    """`e^(ax) cos bx` et `e^(ax) sin bx` : deux parties, puis une equation."""
    exponentielle = sp.exp(A * X)
    cosinus = par_equation(
        (sp.cos(B * X), exponentielle), (-B / A * sp.sin(B * X), exponentielle)
    )
    sinus = par_equation(
        (sp.sin(B * X), exponentielle), (B / A * sp.cos(B * X), exponentielle)
    )
    return cosinus, sinus


def exercice_10() -> Resolution:
    """`racine(x / (1 - x)^3)` : `t^2 = x / (1 - x)`, soit `x = t^2 / (1 + t^2)`."""
    integrande = sp.sqrt(X / (1 - X) ** 3)
    x_de_t, t_de_x = _T**2 / (1 + _T**2), sp.sqrt(X / (1 - X))
    en_t = sp.simplify(integrande.subs(X, x_de_t) * sp.diff(x_de_t, _T))
    primitive = par_changement_inverse(integrande, _T, x_de_t, t_de_x)
    return Resolution(integrande, CHANGEMENT, (en_t,), primitive)


def exercice_11() -> Resolution:
    """`x^2 / racine(1 - x^2)` : `x = sin u`, et il reste `sin^2 u`."""
    en_u = sp.sin(ANGLE) ** 2
    primitive_en_u = (ANGLE - sp.sin(ANGLE) * sp.cos(ANGLE)) / 2
    retour = {
        sp.sin(ANGLE): X,
        sp.cos(ANGLE): sp.sqrt(1 - X**2),
        ANGLE: sp.asin(X),
    }
    primitive = primitive_en_u.subs(retour, simultaneous=True)
    integrande = X**2 / sp.sqrt(1 - X**2)
    return Resolution(integrande, CHANGEMENT, (en_u, primitive_en_u), primitive)


def exercice_12() -> Resolution:
    """`1 / (cos x + 2 sin x + 3)` : `t = tan(x/2)`, puis `1 / ((t + 1)^2 + 1)`."""
    integrande = 1 / (sp.cos(X) + 2 * sp.sin(X) + 3)
    en_t = par_la_tangente_de_l_arc_moitie(integrande)
    primitive = sp.atan(T_DEMI + 1).subs(T_DEMI, sp.tan(X / 2))
    return Resolution(integrande, "tangente de l'arc moitie", (en_t,), primitive)


def primitive_du_12_sur_r() -> sp.Expr:
    """`x/2 + arctan((1 + cos x) / (2 + sin x))` : la meme, continue sur tout R.

    `arctan(tan(x/2) + 1)` saute de `pi` a chaque `x = pi + 2 k pi`. En lui
    retranchant `x/2`, la formule d'addition de l'arc tangente donne un
    quotient dont le denominateur, `2 + sin x`, ne s'annule jamais.
    """
    return X / 2 + sp.atan((1 + sp.cos(X)) / (2 + sp.sin(X)))


def exercice_13() -> Resolution:
    """`racine(x) / racine(a^3 - x^3)` : `u^2 = x^3`, et `du = (3/2) racine(x) dx`."""
    en_u = sp.Rational(2, 3) / sp.sqrt(A**3 - U**2)
    primitive_en_u = sp.Rational(2, 3) * sp.asin(U / A ** sp.Rational(3, 2))
    primitive = primitive_en_u.subs(U, X ** sp.Rational(3, 2))
    integrande = sp.sqrt(X) / sp.sqrt(A**3 - X**3)
    return Resolution(integrande, CHANGEMENT, (en_u, primitive_en_u), primitive)


def exercice_14() -> Resolution:
    """`cosh x / (cosh x + sinh x)` : en exponentielles, `(1 + e^(-2x)) / 2`."""
    integrande = sp.cosh(X) / (sp.cosh(X) + sp.sinh(X))
    en_exponentielles = sp.expand(sp.simplify(integrande.rewrite(sp.exp)))
    primitive = sp.integrate(en_exponentielles, X)
    return Resolution(
        integrande, "passage aux exponentielles", (en_exponentielles,), primitive
    )
