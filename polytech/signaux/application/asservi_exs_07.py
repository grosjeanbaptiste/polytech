"""Feuille « Boucle fermee - stabilite », exercice 1 : un controleur proportionnel.

`K`, puis `1/(p + 1)`, puis `10/(p (2p + 1))`, retour unitaire. La
perturbation entre devant le dernier bloc, avec le signe moins.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.signaux.domain.boucle_fermee import Boucle
from polytech.signaux.domain.stabilite import Stabilite
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import PULSATION as W
from polytech.signaux.domain.transmittance import Transmittance

K: Final = sp.Symbol("K", positive=True)


def boucle() -> Boucle:
    """Le systeme asservi de la figure."""
    return Boucle(
        correcteur=K,
        actionneur=1 / (P + 1),
        procede=10 / (P * (2 * P + 1)),
        capteur=sp.Integer(1),
    )


def _stabilite() -> Stabilite:
    return Stabilite(boucle_ouverte=Transmittance(expression=boucle().ouverte()))


def parties_isochrones() -> tuple[sp.Expr, sp.Expr]:
    """`(a, b)` de `G(j w) = a + j b`, pour `K = 1`."""
    frequentielle = boucle().ouverte().subs({K: 1, P: sp.I * W})
    return sp.simplify(sp.re(frequentielle)), sp.simplify(sp.im(frequentielle))


def asymptote() -> sp.Expr:
    """L'abscisse de l'asymptote verticale : la limite de `a` quand `w` tend vers 0."""
    reelle, _ = parties_isochrones()
    return sp.limit(reelle, W, 0)


def croisement() -> tuple[sp.Expr, sp.Expr]:
    """La pulsation ou la courbe coupe l'axe reel, et l'abscisse pour `K = 1`."""
    etude = _stabilite()
    return etude.pulsation_du_croisement(), etude.reel_au_croisement({K: 1})


def gain_critique_par_nyquist() -> sp.Expr:
    """Le gain qui amene le croisement sur le point `-1`."""
    return _stabilite().gain_critique_par_nyquist(K)


def condition_de_hurwitz() -> sp.Expr:
    """`a_1 a_2 - a_0 a_3`, positif si la boucle fermee est stable."""
    return _stabilite().condition_de_routh()


def gain_critique_par_hurwitz() -> sp.Expr:
    """Le gain qui annule la condition de Hurwitz."""
    return _stabilite().gain_critique(K)


def poles_en_boucle_fermee(gain: object) -> tuple[sp.Expr, ...]:
    """Les racines de `2p^3 + 3p^2 + p + 10 K` pour ce gain."""
    denominateur = sp.fraction(boucle().vis_a_vis_de_la_consigne())[1]
    return tuple(sp.Poly(denominateur.subs(K, gain), P).all_roots())


def poles_instables(gain: object) -> int:
    """Le nombre de poles de la boucle fermee a partie reelle positive."""
    return sum(1 for pole in poles_en_boucle_fermee(gain) if sp.re(pole) > 0)


def est_stable(gain: object) -> bool:
    """Vrai si tous les poles de la boucle fermee sont a gauche."""
    return all(sp.re(pole) < 0 for pole in poles_en_boucle_fermee(gain))
