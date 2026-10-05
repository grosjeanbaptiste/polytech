"""Le produit scalaire et la convolution de signaux analogiques.

Notes de cours de signaux et systemes, sections 1.6.2 et 1.6.3. Le
produit scalaire `<f, g> = integrale de f g*` (1.37) etend aux fonctions
celui des vecteurs ; d'un signal par lui-meme, il rend l'energie. Pour
des signaux periodiques, d'energie infinie, on le rapporte a la periode
(1.39) : il rend alors la puissance.

La convolution `z(t) = integrale de x(tau) y(t - tau)` (1.46) est, pour
chaque `t`, le produit scalaire de `x` par `y` reflechi et decale.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.fourier.transformee import integrale_sur_la_droite
from polytech.signaux.domain.frequence import INSTANT as T

MUETTE = sp.Symbol("tau", real=True)


def produit_scalaire_d_energie(premier: sp.Expr, second: sp.Expr) -> sp.Expr:
    """`<f, g> = integrale sur la droite de f(t) g*(t)`."""
    return sp.simplify(integrale_sur_la_droite(premier * sp.conjugate(second), T))


def produit_scalaire_periodique(
    premier: sp.Expr, second: sp.Expr, periode: sp.Expr
) -> sp.Expr:
    """`(1 / T_0) integrale sur une periode de f(t) g*(t)`."""
    produit = sp.simplify(premier * sp.conjugate(second))
    integrale = sp.integrate(produit, (T, -periode / 2, periode / 2))
    return sp.simplify(integrale / periode)


def _par_le_dirac(integrande: sp.Expr) -> sp.Expr:
    """L'integrale d'un produit par une impulsion : la valeur ou elle agit."""
    impulsions = list(integrande.atoms(sp.DiracDelta))
    if len(impulsions) != 1:
        msg = f"une seule impulsion de Dirac attendue, et non {len(impulsions)}"
        raise ValueError(msg)
    (instant,) = sp.solve(impulsions[0].args[0], MUETTE)
    return sp.simplify((integrande / impulsions[0]).subs(MUETTE, instant))


def convolution_sur_la_droite(premier: sp.Expr, second: sp.Expr) -> sp.Expr:
    """`z(t) = integrale de x(tau) y(t - tau) d tau`."""
    integrande = premier.subs(T, MUETTE) * second.subs(T, T - MUETTE)
    if integrande.has(sp.DiracDelta):
        return _par_le_dirac(integrande)
    return integrale_sur_la_droite(integrande, MUETTE)


def convolution_en(premier: sp.Expr, second: sp.Expr, instant: sp.Expr) -> sp.Expr:
    """`z(t)` en un instant donne : l'aire du produit, reflechi et decale."""
    integrande = premier.subs(T, MUETTE) * second.subs(T, instant - MUETTE)
    return sp.simplify(integrale_sur_la_droite(integrande, MUETTE))
