"""Les quadripoles du TP 3 d'electronique : `R` puis le quadripole `L-C`.

Le premier quadripole est `R` en serie ; le second, `C` en derivation a
son entree puis `L` en serie. Leur cascade, chargee par `R_L`, donne le
transfert ; avec les valeurs de l'enonce, son module au carre prend la
forme d'un filtre de Butterworth du second ordre.
"""

from __future__ import annotations

import sympy as sp

from polytech.electronique.application.symboles_tp_03 import RL, C, L, R, S, W
from polytech.electronique.domain.abcd import (
    cascade,
    derivation,
    gain_en_tension,
    serie,
)

VALEURS = {
    R: 32,
    RL: 32,
    L: 32 / sp.sqrt(2) * sp.Rational(1, 10**5),
    C: sp.Rational(1, 10**5) / (32 * sp.sqrt(2)),
}
"""Les valeurs de l'enonce, exactes : les racines de deux restent symboliques."""


def matrice_de_la_cascade() -> sp.Matrix:
    """`[[1, R], [0, 1]]` fois `[[1, 0], [sC, 1]] [[1, sL], [0, 1]]`."""
    return cascade(serie(R), derivation(1 / (S * C)), serie(S * L))


def transfert_de_la_cascade() -> sp.Expr:
    """`R_L / (A R_L + B)`."""
    return gain_en_tension(matrice_de_la_cascade(), RL)


def filtre_numerique() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """`|H(j omega)|^2`, sa valeur en zero, et la pulsation de coupure."""
    reponse = transfert_de_la_cascade().subs(VALEURS).subs(S, sp.I * W)
    module_carre = sp.simplify(sp.expand(reponse * sp.conjugate(reponse)))
    origine = sp.simplify(module_carre.subs(W, 0))
    (coupure,) = sp.solve(sp.Eq(module_carre, origine / 2), W)
    return module_carre, origine, coupure
