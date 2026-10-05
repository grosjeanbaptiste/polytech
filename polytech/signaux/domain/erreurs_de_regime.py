"""Les erreurs de sortie d'une boucle fermee, en regime etabli.

Planches « Performances » du cours de signaux et systemes. Une boucle
parfaite annulerait l'ecart : la sortie ideale est `Y_i = R / G_s`.
L'erreur est ce qui en separe la sortie reelle,

`E = (T_R - 1 / G_s) R + T_P P = E_C + E_P`,

erreur de consigne et erreur de perturbation. Sous des signaux
constants, et si la boucle est stable, le theoreme de la valeur finale
les donne : `e(infini) = lim p E(p)`.

Sans integrateur dans la chaine, les deux sont finies, d'autant plus
petites que le gain est grand. Un seul integrateur annule l'erreur de
consigne ; il n'annule celle de perturbation que s'il est place avant
elle, dans `C G_1`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P

if TYPE_CHECKING:
    from polytech.signaux.domain.boucle_fermee import Boucle


def sortie_ideale(boucle: Boucle, consigne: sp.Expr) -> sp.Expr:
    """`Y_i = R / G_s` : la sortie qui annulerait l'ecart."""
    return sp.sympify(consigne) / boucle.capteur


def _exiger_la_stabilite(boucle: Boucle) -> None:
    if not boucle.est_stable():
        msg = "boucle instable : le regime etabli n'existe pas"
        raise ValueError(msg)


def erreur_de_consigne(boucle: Boucle, amplitude: sp.Expr) -> sp.Expr:
    """`e_C(infini)` sous une consigne constante `R^0`."""
    _exiger_la_stabilite(boucle)
    ecart = boucle.vis_a_vis_de_la_consigne() - 1 / sp.sympify(boucle.capteur)
    return sp.limit(sp.cancel(ecart) * amplitude, P, 0, "+")


def erreur_de_perturbation(boucle: Boucle, amplitude: sp.Expr) -> sp.Expr:
    """`e_P(infini)` sous une perturbation constante `P^0`."""
    _exiger_la_stabilite(boucle)
    return sp.limit(boucle.vis_a_vis_de_la_perturbation() * amplitude, P, 0, "+")
