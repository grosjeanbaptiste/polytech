"""Les deux circuits de Laplace du TP 3 d'electronique.

**Exercice 2.1** : `C` en serie, `R` en sortie -- un passe-haut. Sa
reponse indicielle et sa reponse a `cos(omega t) u(t)` viennent de
l'inversion de Laplace.

**Exercice 2.2** : `R` et `L` en serie, `C` en sortie -- un passe-bas du
second ordre. Ses poles s'ecrivent `alpha (-1 +- beta)` avec
`alpha = R / (2L)` et `beta = racine(1 - 4L / (C R^2))`, reel ou
imaginaire selon le signe sous la racine. La reponse indicielle a poles
reels se lit sur ses residus ; sans resistance, a la pulsation propre,
le circuit resonne et sa reponse croit lineairement.
"""

from __future__ import annotations

import sympy as sp

from polytech.electronique.application.symboles_tp_03 import C, L, R, S, T, W
from polytech.electronique.domain.laplace import (
    diviseur_de_tension,
    impedance_de_bobine,
    impedance_de_condensateur,
    impedances_en_serie,
)


def _original(transformee: sp.Expr) -> sp.Expr:
    """L'inverse de Laplace pour `t > 0`, l'echelon unite remplace par un."""
    inverse = sp.inverse_laplace_transform(transformee, S, T)
    return sp.simplify(inverse.subs(sp.Heaviside(T), 1))


def transfert_du_2_1() -> sp.Expr:
    """`R / (R + 1/(sC))`."""
    return diviseur_de_tension(impedance_de_condensateur(C, S), R)


def indicielle_du_2_1() -> sp.Expr:
    """La reponse a l'echelon : `e^{-t/(RC)}`."""
    return _original(transfert_du_2_1() / S)


def reponse_au_cosinus_du_2_1() -> sp.Expr:
    """La reponse a `cos(omega t) u(t)` : transitoire plus regime."""
    return _original(transfert_du_2_1() * S / (S**2 + W**2))


def regime_du_2_1() -> sp.Expr:
    """`Re(H(j omega) e^{j omega t})`, la reponse asymptotique par les phaseurs."""
    phaseur = transfert_du_2_1().subs(S, sp.I * W) * sp.exp(sp.I * W * T)
    return sp.simplify(sp.re(sp.expand_complex(phaseur)))


def transfert_du_2_2() -> sp.Expr:
    """`(1/(sC)) / (R + sL + 1/(sC))`."""
    amont = impedances_en_serie(R, impedance_de_bobine(L, S))
    return sp.simplify(diviseur_de_tension(amont, impedance_de_condensateur(C, S)))


def alpha_beta() -> tuple[sp.Expr, sp.Expr]:
    """`alpha = R/(2L)` et `beta = racine(1 - 4L/(C R^2))`."""
    return R / (2 * L), sp.sqrt(1 - 4 * L / (C * R**2))


def poles_du_2_2() -> tuple[sp.Expr, sp.Expr]:
    """`alpha (-1 + beta)` et `alpha (-1 - beta)`."""
    alpha, beta = alpha_beta()
    return alpha * (-1 + beta), alpha * (-1 - beta)


def indicielle_du_2_2() -> sp.Expr:
    """`1 + p2/(p1 - p2) e^{p1 t} + p1/(p2 - p1) e^{p2 t}`, par les residus."""
    p1, p2 = poles_du_2_2()
    return 1 + p2 / (p1 - p2) * sp.exp(p1 * T) + p1 / (p2 - p1) * sp.exp(p2 * T)


def resonance_sans_resistance() -> sp.Expr:
    """`R = 0`, entree `cos(omega_0 t) u(t)` : `(omega_0 t / 2) sin(omega_0 t)`."""
    # Ecrite avec la seule pulsation propre : `1/(LC s^2 + 1)` vaut
    # `w0^2 / (s^2 + w0^2)`, forme que l'inversion symbolique sait traiter.
    propre = sp.Symbol("omega_0", positive=True)
    transformee = propre**2 / (S**2 + propre**2) * S / (S**2 + propre**2)
    return _original(transformee).subs(propre, 1 / sp.sqrt(L * C))
