"""Le coup de belier de juin 2022 : un graphe de fluence du second ordre.

Deux integrateurs, boucles `-alpha` et `-2`, parcours `0` puis `beta` :
`H = beta / (p^2 + alpha p + 2)`. Le parcours nul fait que la sortie ne
saute pas. L'etablissement suit la convention de la fiche : cinq fois la
constante de temps de l'enveloppe, `5 tau = 5 * 2 / alpha`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from scipy.optimize import brentq

from polytech.signaux.domain.fluence import transmittance_du_graphe

if TYPE_CHECKING:
    import sympy as sp

RAIDEUR = 2
"""Le `-2` du graphe : `omega_0^2`."""
CONSTANTES_DE_TEMPS = 5


def graphe_du_coup_de_belier(alpha: sp.Expr, beta: sp.Expr) -> sp.Expr:
    """`H(p)` du graphe, par la regle de Mason."""
    return transmittance_du_graphe(retours=[-alpha, -RAIDEUR], directs=[0, beta])


def reglage_pour_un_etablissement(periodes: float) -> float:
    """L'`alpha` qui etablit la sortie en `periodes` pseudo-periodes."""
    if periodes <= 0:
        msg = f"nombre de periodes non positif : {periodes}"
        raise ValueError(msg)
    limite = 2 * math.sqrt(RAIDEUR)

    def ecart(alpha: float) -> float:
        etablissement = CONSTANTES_DE_TEMPS * 2 / alpha
        pseudo = 2 * math.pi / math.sqrt(RAIDEUR - alpha**2 / 4)
        return etablissement - periodes * pseudo

    return float(brentq(ecart, 1e-6, limite - 1e-9))
