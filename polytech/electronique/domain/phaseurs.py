"""Les phaseurs : un cosinus resume par un nombre complexe.

Planches du chapitre 1, partie III (planches 49 a 58).

    A cos(w t + phi) = Re[ A e^(j phi) e^(j w t) ],

et le **phaseur** est `A e^(j phi)`. Deriver le signal multiplie son
phaseur par `j w` : la bobine devient `V = j w L I`, le condensateur `V =
I / (j w C)`, et un circuit lineaire une fonction de transfert `H(j w)`.

Excite par un cosinus, un circuit stable repond par un cosinus de meme
pulsation, auquel s'ajoutent des termes qui s'eteignent. Chercher un
phaseur, c'est ne garder que ce qui reste : `A |H(j w)| cos(w t + phi +
arg H(j w))`. L'analyse ne dit rien du transitoire.
"""

from __future__ import annotations

import math

import sympy as sp


def phaseur(amplitude: sp.Expr, phase: sp.Expr) -> sp.Expr:
    """`A e^(j phi)`, avec `A >= 0`."""
    if sp.sympify(amplitude).is_negative:
        msg = f"amplitude negative : {amplitude}"
        raise ValueError(msg)
    return amplitude * sp.exp(sp.I * phase)


def signal_du_phaseur(complexe: sp.Expr, pulsation: sp.Expr, temps: sp.Expr) -> sp.Expr:
    """`Re[P e^(j w t)]` : le signal que le phaseur represente."""
    tournant = sp.expand_complex(complexe * sp.exp(sp.I * pulsation * temps))
    return sp.simplify(sp.re(tournant))


def somme_de_sinus(
    premiere: float, phase_a: float, seconde: float, phase_b: float
) -> tuple[float, float]:
    """`a sin(wt + theta_a) + b sin(wt + theta_b) = c sin(wt + phi)` : `(c, phi)`."""
    reelle = premiere * math.cos(phase_a) + seconde * math.cos(phase_b)
    imaginaire = premiere * math.sin(phase_a) + seconde * math.sin(phase_b)
    return math.hypot(reelle, imaginaire), math.atan2(imaginaire, reelle)


def reponse_asymptotique(
    transfert: sp.Expr,
    variable: sp.Symbol,
    pulsation: sp.Expr,
    *,
    amplitude: sp.Expr | int = 1,
    phase: sp.Expr | int = 0,
) -> tuple[sp.Expr, sp.Expr]:
    """L'amplitude et la phase de la sortie : `A |H(j w)|` et `phi + arg H(j w)`."""
    harmonique = transfert.subs(variable, sp.I * pulsation)
    module = sp.simplify(amplitude * sp.Abs(harmonique))
    return module, sp.simplify(phase + sp.arg(harmonique))
