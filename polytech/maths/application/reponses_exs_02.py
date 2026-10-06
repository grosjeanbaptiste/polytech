"""Mathematique, exercices 2, II.6 et II.7 : reponses indicielle et impulsionnelle.

Source : `Mathematique/EXS/Exs_02_Equations_differentielles.pdf`, pages 14
et 15.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.differentielles.operateurs import D, OperateurDifferentiel
from polytech.maths.domain.laplace.green import (
    conditions_initiales_equivalentes,
    fonction_de_green,
    reponse,
    reponse_au_creneau,
)
from polytech.maths.domain.laplace.heaviside import par_fractions_simples
from polytech.maths.domain.laplace.ordre_n import ProblemeLineaire
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T

HAUTEUR: Final = sp.Symbol("n", positive=True)
OPERATEUR: Final = OperateurDifferentiel(D**2 + 4 * D + 3)
"""II.6 : `y'' + 4y' + 3y`."""
RESISTANCE: Final = sp.Integer(3)
INDUCTANCE: Final = sp.Integer(1)
CAPACITE: Final = sp.Rational(1, 2)
"""II.7 : `R = 3` ohms, `L = 1` H, `C = 0,5` F."""


def indicielle() -> sp.Expr:
    """II.6.1 : la reponse a l'echelon, au repos."""
    return reponse(OPERATEUR, sp.Integer(1))


def au_creneau() -> tuple[sp.Expr, sp.Expr]:
    """II.6.2 : la reponse au creneau de hauteur `n`, pendant puis apres."""
    return reponse_au_creneau(OPERATEUR, HAUTEUR)


def impulsionnelle() -> sp.Expr:
    """II.6.3 : `h`, l'original de `1/K(p)`."""
    return fonction_de_green(OPERATEUR)


def conditions_equivalentes() -> tuple[sp.Expr, ...]:
    """Les conditions initiales qui font de `h` une reponse libre."""
    return conditions_initiales_equivalentes(OPERATEUR)


def libre() -> sp.Expr:
    """II.6.4 : la reponse sans sollicitation a `y(0) = 0`, `y'(0) = 1`."""
    return ProblemeLineaire((1, 4, 3), sp.Integer(0), (0, 1)).solution()


def admittance() -> sp.Expr:
    """II.7 : `I/U = 1/(L p + R + 1/(C p))`, condensateur decharge et `i(0) = 0`."""
    impedance = INDUCTANCE * P + RESISTANCE + 1 / (CAPACITE * P)
    return sp.simplify(1 / impedance)


def courant_sinusoidal(pulsation: sp.Expr) -> sp.Expr:
    """Le courant pour `u = sin(omega t)`."""
    tension = pulsation / (P**2 + pulsation**2)
    brut = par_fractions_simples(admittance() * tension)
    return sp.simplify(sp.re(sp.expand_complex(brut)))


def courants() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Les reponses a l'echelon, a l'impulsion, et a `sin t`."""
    a_l_echelon = par_fractions_simples(
        admittance() * Transformee.de(sp.Integer(1)).image
    )
    a_l_impulsion = par_fractions_simples(admittance())
    return a_l_echelon, a_l_impulsion, courant_sinusoidal(sp.Integer(1))


def residu_en_j() -> sp.Expr:
    """Le coefficient de `e^(jt)` dans le courant sinusoidal : le residu en `p = j`."""
    variable = sp.Symbol("s")
    image = (admittance() / (P**2 + 1)).subs(P, variable)
    return sp.simplify(sp.residue(image, variable, sp.I))


def regime_etabli() -> tuple[sp.Expr, sp.Expr]:
    """L'amplitude et l'expression du courant etabli, par l'impedance en `p = j`."""
    complexe = sp.simplify(admittance().subs(P, sp.I))
    etabli = sp.re(complexe) * sp.sin(T) + sp.im(complexe) * sp.cos(T)
    return sp.Abs(complexe), etabli


def resonance_du_circuit() -> tuple[sp.Expr, sp.Expr]:
    """La pulsation ou l'amplitude etablie est maximale, et ce maximum."""
    pulsation = sp.Symbol("omega", positive=True)
    carre = sp.simplify(sp.Abs(admittance().subs(P, sp.I * pulsation)) ** 2)
    (optimale,) = sp.solve(sp.diff(carre, pulsation), pulsation)
    return optimale, sp.sqrt(carre.subs(pulsation, optimale))
