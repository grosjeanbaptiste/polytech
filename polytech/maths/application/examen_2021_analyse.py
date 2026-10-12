r"""Examen de janvier 2021 : primitives, equations du premier ordre, recurrences.

Source : `Mathematique/EXAMENS/Exam_2021_Janvier_Enonce.pdf`, questions 1, 3
et 4, trois options chacune. Chaque fonction rend l'expression que le
corrige compose ; les tests les recoupent par un autre chemin.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.differentielles.lineaire import EquationLineaire
from polytech.maths.domain.differentielles.premier_ordre import VARIABLE as T
from polytech.maths.domain.integration.fractions import FractionRationnelle
from polytech.maths.domain.integration.parties import IntegrationParParties
from polytech.maths.domain.symboles import CONSTANTE as C
from polytech.maths.domain.symboles import VARIABLE as X

EQUATIONS: Final = {
    7: (EquationLineaire(-2 * T, T), sp.Integer(1), sp.E - sp.Rational(1, 2)),
    8: (EquationLineaire(sp.tan(T), sp.sin(2 * T)), sp.Integer(0), sp.Integer(6)),
    9: (
        EquationLineaire(sp.cot(T), 5 * sp.exp(sp.cos(T))),
        sp.pi / 2,
        sp.Integer(-4),
    ),
}
"""Numero de l'enonce : l'equation, l'instant de la condition, sa valeur."""


def etape_i() -> sp.Expr:
    """`x tan(x/2) - int tan(x/2) dx` : `u = x`, `dv = dx / (2 cos^2(x/2))`."""
    parties = IntegrationParParties(X, 1 / (2 * sp.cos(X / 2) ** 2))
    return X * sp.tan(X / 2) - sp.Integral(sp.tan(X / 2) * parties.derivee_de_u(), X)


def primitive_i() -> sp.Expr:
    """`x tan(x/2) + 2 ln(cos(x/2))`, la tangente integree en logarithme."""
    return X * sp.tan(X / 2) + 2 * sp.log(sp.cos(X / 2))


def termes_j() -> tuple[sp.Expr, ...]:
    """Les fractions simples de `(1 + 2x - 3x^2) / (x - x^3)`."""
    return FractionRationnelle(1 + 2 * X - 3 * X**2, X - X**3).termes()


def primitive_j() -> sp.Expr:
    """La primitive de la somme des fractions simples."""
    return FractionRationnelle(1 + 2 * X - 3 * X**2, X - X**3).primitive()


def primitive_k() -> sp.Expr:
    """`-ln|1 - ln^2 x| / 2`, apres `xi = ln x` ; vaut des deux cotes de `e`."""
    xi = sp.Symbol("xi", real=True)
    return (-sp.log(sp.Abs(1 - xi**2)) / 2).subs(xi, sp.log(X))


def generale_de(numero: int) -> sp.Expr:
    """La solution generale de l'equation ; leve si l'enonce n'en a pas de ce numero."""
    if numero not in EQUATIONS:
        msg = f"pas d'equation {numero} a la question 3"
        raise KeyError(msg)
    return sp.simplify(EQUATIONS[numero][0].solution_generale())


def solution_de(numero: int) -> sp.Expr:
    """La solution qui satisfait la condition de l'enonce."""
    generale = generale_de(numero)
    _, instant, valeur = EQUATIONS[numero]
    (constante,) = sp.solve(sp.Eq(generale.subs(T, instant), valeur), C)
    return sp.simplify(generale.subs(C, constante))


def i_deux() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """`I_2` des integrales (10), (11) et (12), chacune par sa recurrence."""
    dix = sp.Rational(1, 2) * sp.atan(1) + sp.Rational(1, 4)
    un = X * sp.log(X) - X
    onze = X * sp.log(X) ** 2 - 2 * un
    douze = (sp.sqrt(2) - sp.asinh(1).rewrite(sp.log)) / 2
    return dix, sp.expand(onze), douze
