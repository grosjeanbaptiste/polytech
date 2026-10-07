"""Recueil d'exercices, planche 16 : la trottinette et la bordure.

Le graphe a deux integrateurs, de retours `-gamma` et `-1`, et un seul
gain direct, `beta`, sur le second. La bordure est un echelon ; on regle
`beta` et `gamma` pour que le plancher la suive apres une oscillation et
demie.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.signaux.domain.fluence import transmittance_du_graphe
from polytech.signaux.domain.symboles import IMAGE as P

BETA, GAMMA = sp.symbols("beta gamma", positive=True)
OSCILLATIONS: Final = sp.Rational(3, 2)
CONSTANTES_POUR_S_ETEINDRE: Final = 5
HAUTEUR_FINALE: Final = 1
"""Le plancher finit a la hauteur de la bordure."""


def transmittance() -> sp.Expr:
    """`H(p)` du graphe, par la regle de Mason."""
    return transmittance_du_graphe(retours=(-GAMMA, -1), directs=(0, BETA))


def depart_de_l_indicielle() -> tuple[sp.Expr, sp.Expr]:
    """`a(0+)` et `a'(0+)` : les limites de `H(p)` et de `p H(p)` a l'infini."""
    h = transmittance()
    return sp.limit(h, P, sp.oo), sp.limit(P * h, P, sp.oo)


def beta() -> sp.Expr:
    """Le gain pour lequel `a(infini) = H(0)` vaut la hauteur de la bordure."""
    (solution,) = sp.solve(sp.Eq(transmittance().subs(P, 0), HAUTEUR_FINALE), BETA)
    return solution


def _duree_et_periode() -> tuple[sp.Expr, sp.Expr]:
    """Cinq constantes de temps, et la periode des oscillations amorties."""
    sigma = GAMMA / 2
    return CONSTANTES_POUR_S_ETEINDRE / sigma, 2 * sp.pi / sp.sqrt(1 - sigma**2)


def gamma_exact() -> sp.Expr:
    """Le `gamma` pour lequel cinq constantes de temps font 1,5 periode."""
    duree, periode = _duree_et_periode()
    (solution,) = sp.solve(sp.Eq(duree, OSCILLATIONS * periode), GAMMA)
    return sp.simplify(solution)


def gamma_approche() -> sp.Expr:
    """Celui du manuscrit, qui prend `3 pi / 10` pour un."""
    (solution,) = sp.solve(sp.Eq(GAMMA**2, 1 - GAMMA**2 / 4), GAMMA)
    return solution


def oscillations_avant_le_regime(gamma: sp.Expr) -> float:
    """Le nombre de periodes que durent cinq constantes de temps."""
    duree, periode = _duree_et_periode()
    return float((duree / periode).subs(GAMMA, gamma))
