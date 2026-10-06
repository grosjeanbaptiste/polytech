"""Mathematique, exercices 2, II.3 a II.5 : valeur finale, convolution, equations.

Source : `Mathematique/EXS/Exs_02_Equations_differentielles.pdf`, pages 12
et 13.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.laplace.convolution import produit_de_convolution
from polytech.maths.domain.laplace.heaviside import (
    fractions_simples,
    par_fractions_simples,
)
from polytech.maths.domain.laplace.ordre_n import ProblemeLineaire
from polytech.maths.domain.laplace.proprietes import de_la_convolution
from polytech.maths.domain.laplace.theoremes import (
    limite_en_zero,
    poles_genants,
    valeur_finale,
)
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T

A: Final = sp.Symbol("a", positive=True)
"""Le parametre de II.3, que l'enonce declare positif."""
STABLE: Final = sp.Integer(-2)
"""Une valeur negative de `a`, pour laquelle le theoreme s'applique."""
FACTEURS: Final = (T, sp.sin(T) - sp.exp(T))
"""II.4 : les deux fonctions dont on prend le produit de convolution."""


def image_du_3(parametre: sp.Expr = A) -> sp.Expr:
    """II.3 : `F(p) = 1/(p (p - a)^2)`."""
    return 1 / (P * (P - parametre) ** 2)


def valeur_par_la_formule() -> sp.Expr:
    """`lim p F(p)` en zero : ce que la formule rend, applicable ou non."""
    return limite_en_zero(image_du_3())


def poles_a_droite() -> tuple[sp.Expr, ...]:
    """Les poles de `p F(p)` qui interdisent le theoreme."""
    return poles_genants(image_du_3())


def valeur_finale_par_le_theoreme() -> sp.Expr:
    """La limite de `f`, si le theoreme s'applique ; il leve sinon."""
    return valeur_finale(image_du_3())


def originale_du_3(parametre: sp.Expr = A) -> sp.Expr:
    """L'originale de `F`, par fractions simples."""
    return par_fractions_simples(image_du_3(parametre))


def cas_stable() -> tuple[sp.Expr, sp.Expr]:
    """Avec `a < 0` : la limite de l'originale, et celle que le theoreme donne."""
    limite = sp.limit(originale_du_3(STABLE), T, sp.oo)
    return limite, valeur_finale(image_du_3(STABLE))


def image_de_la_convolution() -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    """II.4 : le produit des deux images, et ses fractions simples."""
    premiere, seconde = (Transformee.de(f).image for f in FACTEURS)
    image = de_la_convolution(premiere, seconde)
    return image, fractions_simples(image)


def convolution() -> tuple[sp.Expr, sp.Expr]:
    """II.4 : le produit par son image, puis par l'integrale elle-meme."""
    image, _ = image_de_la_convolution()
    return par_fractions_simples(image), produit_de_convolution(*FACTEURS)


def probleme_1() -> ProblemeLineaire:
    """II.5.1 : `y'' - 5y' + 6y = 0`, `y(0) = 1`, `y'(0) = 1`."""
    return ProblemeLineaire((1, -5, 6), sp.Integer(0), (1, 1))


def probleme_2() -> ProblemeLineaire:
    """II.5.2 : `y'' + 2y' + y = 4 e^(-t)`, `y(0) = 2`, `y'(0) = -1`."""
    return ProblemeLineaire((1, 2, 1), 4 * sp.exp(-T), (2, -1))


def rampe_ecretee() -> tuple[sp.Expr, sp.Expr]:
    """II.5.3 : `y'' + 4y = t - tau_1(t)`, au repos ; avant `t = 1`, puis apres.

    Le second membre est une rampe moins la meme rampe retardee de 1.
    L'image de la reponse a la rampe est `1/(p^2 (p^2 + 4))` ; le theoreme
    du retard donne le reste.
    """
    a_la_rampe = par_fractions_simples(Transformee.de(T).image / (P**2 + 4))
    return a_la_rampe, sp.simplify(a_la_rampe - a_la_rampe.subs(T, T - 1))
