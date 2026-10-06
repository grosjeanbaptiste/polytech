"""Feuille « Boucle fermee - stabilite », exercice 2 : Nyquist, puis un gain.

`T(p) = (1 + 2p)/(p (1 + 5p))` : un integrateur, un zero, un pole. Sa
courbe de Nyquist ne coupe jamais l'axe reel, et c'est tout l'exercice.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.signaux.domain.stabilite import Stabilite, equation_caracteristique
from polytech.signaux.domain.stabilite_des_poles import nature_de_la_stabilite
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import PULSATION as W
from polytech.signaux.domain.transmittance import Transmittance

K: Final = sp.Symbol("K", positive=True)
TRANSMITTANCE: Final = (1 + 2 * P) / (P * (1 + 5 * P))


def parties_isochrones() -> tuple[sp.Expr, sp.Expr]:
    """`(a, b)` de `T(j w) = a + j b`."""
    frequentielle = TRANSMITTANCE.subs(P, sp.I * W)
    return sp.simplify(sp.re(frequentielle)), sp.simplify(sp.im(frequentielle))


def asymptote() -> sp.Expr:
    """L'abscisse de l'asymptote verticale, quand `w` tend vers zero."""
    reelle, _ = parties_isochrones()
    return sp.limit(reelle, W, 0)


def coupe_l_axe_reel() -> bool:
    """Faux : la partie imaginaire ne s'annule a aucune pulsation positive."""
    etude = Stabilite(boucle_ouverte=Transmittance(expression=TRANSMITTANCE))
    try:
        etude.pulsation_du_croisement()
    except ValueError:  # le domaine leve quand il n'y a pas de croisement
        return False
    return True


def nature_en_boucle_ouverte() -> str:
    """Ce que les poles de `T` disent de sa stabilite propre."""
    return nature_de_la_stabilite(TRANSMITTANCE)


def equation_caracteristique_de_la_boucle() -> sp.Expr:
    """`1 + K T(p) = 0`, mise en polynome."""
    ouverte = Transmittance(expression=K * TRANSMITTANCE)
    return sp.collect(equation_caracteristique(ouverte), P)


def poles_en_boucle_fermee(gain: object) -> tuple[sp.Expr, ...]:
    """Les racines de l'equation caracteristique pour ce gain."""
    polynome = equation_caracteristique_de_la_boucle().subs(K, gain)
    return tuple(sp.Poly(polynome, P).all_roots())


def est_stable(gain: object) -> bool:
    """Vrai si les deux poles de la boucle fermee sont a gauche."""
    return all(sp.re(pole) < 0 for pole in poles_en_boucle_fermee(gain))
