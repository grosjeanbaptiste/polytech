"""Feuille « Systemes », exercices 5 et 6 : deux transmittances a etablir.

Une masse tiree par un chariot a travers un ressort et un amortisseur ;
un condensateur en parallele sur une bobine resistive, alimente en
courant. Les deux sont du second ordre, et leurs equations se
ressemblent terme a terme.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from polytech.signaux.domain.modeles import coefficients
from polytech.signaux.domain.poles_numeriques import reponse_indicielle
from polytech.signaux.domain.symboles import IMAGE
from polytech.signaux.domain.transmittance import Transmittance

if TYPE_CHECKING:
    from numpy.typing import NDArray

M, C, K = sp.symbols("m c k", positive=True)
R, L, CAPACITE = sp.symbols("R L C", positive=True)
EXEMPLE: Final = {M: 2, C: 3, K: 5}
"""Des valeurs quelconques, pour comparer une simulation a la transmittance."""


def transmittance_du_chariot() -> Transmittance:
    """`Y/U` : Newton sur la masse, `m y'' = k (u - y) + c (u' - y')`."""
    sortie, entree = sp.symbols("Y U")
    newton = sp.Eq(M * IMAGE**2 * sortie, (K + C * IMAGE) * (entree - sortie))
    (solution,) = sp.solve(newton, sortie)
    return Transmittance(expression=sp.cancel(solution / entree))


def reponses_du_chariot(
    instants: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """La reponse a un echelon de `u` : Newton integre, puis la transmittance.

    L'echelon rend `u'` impulsionnel : on integre `z = y' - (c/m) u`.
    """
    masse, frottement, raideur = (float(EXEMPLE[s]) for s in (M, C, K))

    def derivees(_: float, etat: NDArray[np.float64]) -> list[float]:
        position, auxiliaire = etat
        vitesse = auxiliaire + frottement / masse
        return [vitesse, (raideur * (1 - position) - frottement * vitesse) / masse]

    bornes = (float(instants[0]), float(instants[-1]))
    solution = solve_ivp(
        derivees, bornes, [0.0, 0.0], t_eval=instants, rtol=1e-11, atol=1e-13
    )
    chiffree = Transmittance(
        expression=transmittance_du_chariot().expression.subs(EXEMPLE)
    )
    numerateur, denominateur = coefficients(chiffree)
    analytique = reponse_indicielle(numerateur, denominateur, instants)
    return np.asarray(solution.y[0], dtype=float), analytique


def transmittance_du_circuit() -> Transmittance:
    """`V_c/I` : la loi des noeuds, `I = C p V_c + V_c / (R + L p)`."""
    tension, courant = sp.symbols("V I")
    noeud = sp.Eq(courant, CAPACITE * IMAGE * tension + tension / (R + L * IMAGE))
    (solution,) = sp.solve(noeud, tension)
    return Transmittance(expression=sp.cancel(solution / courant))
