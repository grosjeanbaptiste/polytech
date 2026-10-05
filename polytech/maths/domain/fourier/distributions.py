"""Les distributions : une transformee de Fourier pour la constante et le cosinus.

Une **distribution** est une fonctionnelle lineaire et continue sur un
espace de fonctions test. La plus simple, `delta_a`, evalue la fonction :
`<delta_a, phi> = phi(a)`.

Les fonctions test de `D(R)` sont lisses et a support compact ; leur
transformee de Fourier ne l'est plus. On les elargit donc a l'espace de
Schwartz `S(R)` des fonctions lisses **a decroissance rapide** -- tous les
`x^n phi^(p)` y sont bornes --, stable par Fourier, et l'on definit la
transformee d'une distribution temperee par `<T^, phi> = <T, phi^>`. Alors

    1            a pour image  2 pi delta(w),
    cos(w0 x)    a pour image  pi (delta(w - w0) + delta(w + w0)),
    delta_a      a pour image  e^(-j w a).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.domain.fourier.transformee import PULSATION
from polytech.maths.domain.symboles import VARIABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

FONCTION_TEST: Final = sp.Piecewise(
    (sp.exp(1 / (VARIABLE**2 - 1)), sp.Abs(VARIABLE) < 1), (0, True)
)
"""`exp(1 / (x^2 - 1))` sur `]-1, 1[`, nulle ailleurs : lisse, a support compact."""


def appliquer_le_dirac(point: object, fonction: sp.Expr) -> sp.Expr:
    """`<delta_a, phi> = phi(a)`."""
    return sp.sympify(fonction).subs(VARIABLE, sp.sympify(point))


def appliquer_le_peigne(
    periode: object, demi_nombre: int, fonction: sp.Expr
) -> sp.Expr:
    """Le peigne `somme delta(x - nT)`, tronque a `|n| <= demi_nombre`."""
    pas = sp.sympify(periode)
    dents = range(-demi_nombre, demi_nombre + 1)
    return sum((appliquer_le_dirac(n * pas, fonction) for n in dents), sp.Integer(0))


def est_a_decroissance_rapide(fonction: sp.Expr, puissance: int, ordre: int) -> bool:
    """Vrai si `x^n phi^(p)(x)` reste borne quand `x` tend vers l'infini."""
    produit = VARIABLE**puissance * sp.diff(sp.sympify(fonction), VARIABLE, ordre)
    limites = (sp.limit(produit, VARIABLE, sens) for sens in (sp.oo, -sp.oo))
    return all(limite.is_finite for limite in limites)


def transformee_de_la_constante(valeur: object) -> sp.Expr:
    """`2 pi c delta(w)`."""
    return 2 * sp.pi * sp.sympify(valeur) * sp.DiracDelta(PULSATION)


def transformee_du_cosinus(pulsation: object) -> sp.Expr:
    """`pi (delta(w - w0) + delta(w + w0))`."""
    w0 = sp.sympify(pulsation)
    return sp.pi * (sp.DiracDelta(PULSATION - w0) + sp.DiracDelta(PULSATION + w0))


def transformee_du_sinus(pulsation: object) -> sp.Expr:
    """`-j pi (delta(w - w0) - delta(w + w0))`."""
    w0 = sp.sympify(pulsation)
    pics = sp.DiracDelta(PULSATION - w0) - sp.DiracDelta(PULSATION + w0)
    return -sp.I * sp.pi * pics


def transformee_des_impulsions(positions: Sequence[object]) -> sp.Expr:
    """`somme e^(-j w a)`, l'image de `somme delta_a`."""
    return sum(
        (sp.exp(-sp.I * PULSATION * sp.sympify(a)) for a in positions), sp.Integer(0)
    )
