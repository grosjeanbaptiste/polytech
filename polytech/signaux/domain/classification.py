"""La caracterisation d'un signal : parite, causalite, puissance.

Notes de cours de signaux et systemes, section 1.2. Un signal est causal
s'il est nul aux instants negatifs. Il se decompose toujours en une
partie paire et une partie impaire, dont la somme le reconstitue.

Un signal periodique est d'energie infinie ; sa puissance se calcule sur
une periode (1.4), et sa valeur efficace en est la racine (1.6) :
l'amplitude du signal constant qui aurait la meme puissance.
"""

from __future__ import annotations

import sympy as sp

from polytech.signaux.domain.frequence import INSTANT as T


def partie_paire(signal: sp.Expr) -> sp.Expr:
    """`(f(t) + f(-t)) / 2`."""
    return (signal + signal.subs(T, -T)) / 2


def partie_impaire(signal: sp.Expr) -> sp.Expr:
    """`(f(t) - f(-t)) / 2`."""
    return (signal - signal.subs(T, -T)) / 2


def est_causal(signal: sp.Expr) -> bool:
    """Vrai si le signal est identiquement nul pour `t < 0`."""
    avant = sp.Symbol("u", negative=True)
    return bool(sp.simplify(signal.subs(T, avant)) == 0)


def puissance_sur_une_periode(signal: sp.Expr, periode: sp.Expr) -> sp.Expr:
    """`P = (1 / T_0) integrale de |f|^2` sur une periode."""
    if not sp.sympify(periode).is_positive:
        msg = f"periode nulle ou negative : {periode}"
        raise ValueError(msg)
    carre = sp.simplify(signal * sp.conjugate(signal))
    integrale = sp.integrate(carre, (T, -periode / 2, periode / 2))
    return sp.simplify(integrale / periode)


def valeur_efficace_sur_une_periode(signal: sp.Expr, periode: sp.Expr) -> sp.Expr:
    """`F_RMS`, la racine de la puissance."""
    return sp.sqrt(puissance_sur_une_periode(signal, periode))
