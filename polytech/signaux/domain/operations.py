"""Les operations elementaires sur les signaux.

Notes de cours de signaux et systemes, section 1.6.1. Un signal etant
une fonction du temps, on le decale (`f(t - t_0)`), on le reflechit
(`f(-t)`), on le contracte ou le dilate (`f(alpha t)`).

Deux sinusoides de meme frequence s'ajoutent en une sinusoide de cette
frequence, dont l'amplitude et la phase sont celles de la somme des
phaseurs (1.35). Leur produit, lui, change les frequences : il rend la
somme et la difference des deux. Et repeter un signal tous les `T_0` le
rend periodique (1.36).
"""

from __future__ import annotations

import sympy as sp

from polytech.signaux.domain.frequence import INSTANT as T

type Phaseur = tuple[sp.Expr, sp.Expr]
"""Une amplitude et une phase initiale."""


def decaler(signal: sp.Expr, retard: sp.Expr) -> sp.Expr:
    """`f(t - t_0)`."""
    return signal.subs(T, T - retard)


def reflechir(signal: sp.Expr) -> sp.Expr:
    """`f(-t)`."""
    return signal.subs(T, -T)


def dilater(signal: sp.Expr, facteur: sp.Expr) -> sp.Expr:
    """`f(alpha t)` : contracte si `alpha > 1`, dilate si `alpha < 1`."""
    if facteur == 0:
        msg = "facteur d'echelle nul : le signal n'aurait plus d'axe du temps"
        raise ValueError(msg)
    return signal.subs(T, facteur * T)


def somme_de_sinusoides(premiere: Phaseur, seconde: Phaseur) -> Phaseur:
    """L'amplitude et la phase de `a_1 cos(w t + phi_1) + a_2 cos(w t + phi_2)`."""
    somme = sum(
        (a * sp.exp(sp.I * phase) for a, phase in (premiere, seconde)), sp.S.Zero
    )
    return (sp.simplify(sp.Abs(somme)), sp.simplify(sp.arg(somme)))


def produit_de_cosinus(premiere: sp.Expr, seconde: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Les pulsations somme et difference de `cos(w_1 t) cos(w_2 t)`."""
    return (premiere + seconde, sp.Abs(premiere - seconde))


def periodifier(signal: sp.Expr, periode: sp.Expr, *, repliques: int) -> sp.Expr:
    """`somme de f(t - k T_0)`, pour `k` de `-repliques` a `repliques`."""
    if not sp.sympify(periode).is_positive:
        msg = f"periode nulle ou negative : {periode}"
        raise ValueError(msg)
    rangs = range(-repliques, repliques + 1)
    return sum((decaler(signal, rang * periode) for rang in rangs), sp.S.Zero)
