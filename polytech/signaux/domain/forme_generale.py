"""La forme generale de `H(p)` et de `h(t)`.

Syllabus de signaux et systemes, sections 4.1.4 et 4.1.7. Une fraction
rationnelle est definie, a un facteur pres, par ses poles et ses zeros
(4.11) : `H(p) = K produit(p - z_i) / produit(p - p_i)`.

La reponse a un signal reel etant reelle, les poles sont reels ou vont
par paires conjuguees `-sigma +/- j omega`, de residus conjugues. Les
deux exponentielles complexes d'une paire se regroupent (4.21) :

`A exp(p t) + A* exp(p* t) = 2 |A| exp(-sigma t) cos(omega t + arg A)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import TEMPS as T
from polytech.signaux.domain.transmittance import Transmittance

if TYPE_CHECKING:
    from collections.abc import Sequence


def forme_factorisee(
    gain: sp.Expr, *, zeros: Sequence[sp.Expr], poles: Sequence[sp.Expr]
) -> Transmittance:
    """`K produit(p - z_i) / produit(p - p_i)`."""
    if not poles:
        msg = "aucun pole : ce n'est pas la transmittance d'un systeme dynamique"
        raise ValueError(msg)
    numerateur = sp.Mul(*(P - zero for zero in zeros))
    denominateur = sp.Mul(*(P - pole for pole in poles))
    return Transmittance(expression=gain * numerateur / denominateur)


def mode_d_une_paire_conjuguee(residu: sp.Expr, pole: sp.Expr) -> sp.Expr:
    """`2 |A| exp(-sigma t) cos(omega t + arg A)`."""
    amplitude, pulsation = 2 * sp.Abs(residu), sp.im(pole)
    return amplitude * sp.exp(sp.re(pole) * T) * sp.cos(pulsation * T + sp.arg(residu))
