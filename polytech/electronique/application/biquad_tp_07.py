"""TP 7 d'electronique, exercice 2.1 : le biquad passe-bas de gain unite.

`H(s) = w_0^2 / (s^2 + (w_0 / Q) s + w_0^2)`. Les limites sont prises par
le calcul formel, les poles sortent du denominateur, et la figure du
corrige officiel (page 4) est retracee : cinq amortissements, pour
`w_0 = 1000 rad/s`.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import sympy as sp

from polytech.electronique.domain.reponse_du_biquad import (
    coupure_a_trois_decibels,
    module_carre,
)
from polytech.electronique.domain.second_ordre import PasseBasCanonique

S: Final = sp.Symbol("s", nonzero=True)
W0: Final = sp.Symbol("omega_0", positive=True)
_Q: Final = sp.Symbol("Q", positive=True)

PULSATION_DE_LA_FIGURE: Final = 1000.0
AMORTISSEMENTS: Final = (0.02, 0.1, 0.3, 0.5, 1.0)
"""Les cinq `xi` de la figure du corrige officiel."""
POINTS_DE_LA_FIGURE: Final = 241


def limites_du_transfert() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """`H(0)`, puis `H(s)` hors de l'origine pour `Q -> 0+` et `Q -> infini`."""
    transfert = PasseBasCanonique(sp.Integer(1), W0, _Q).transfert(S)
    a_l_origine = PasseBasCanonique(sp.Integer(1), W0, _Q).gain_continu()
    return (
        a_l_origine,
        sp.limit(transfert, _Q, 0, "+"),
        sp.limit(transfert, _Q, sp.oo),
    )


def poles_reduits(amortissement: float) -> tuple[complex, complex]:
    """Les deux poles pour `w_0 = 1`, en fonction de `xi = 1 / (2 Q)`."""
    if amortissement <= 0:
        msg = f"amortissement nul ou negatif : {amortissement}"
        raise ValueError(msg)
    qualite = 1 / (2 * sp.nsimplify(amortissement, rational=True))
    premier, second = PasseBasCanonique(sp.Integer(1), sp.Integer(1), qualite).poles()
    return complex(premier), complex(second)


def gain_en_decibels(pulsation: float, amortissement: float) -> float:
    """`20 log |H(j w)|` pour la pulsation propre de la figure."""
    reduite = pulsation / PULSATION_DE_LA_FIGURE
    return 10 * math.log10(module_carre(reduite, 1 / (2 * amortissement)))


def coupure_de_la_figure() -> float:
    """La coupure a 3 dB de la courbe la moins amortie, en rad/s."""
    qualite = 1 / (2 * AMORTISSEMENTS[0])
    return PULSATION_DE_LA_FIGURE * coupure_a_trois_decibels(qualite)


def courbe(amortissement: float) -> tuple[np.ndarray, np.ndarray]:
    """Les pulsations, de 100 a 10 000 rad/s, et le gain en decibels."""
    pulsations = np.logspace(2, 4, POINTS_DE_LA_FIGURE)
    gains = np.array([gain_en_decibels(float(w), amortissement) for w in pulsations])
    return pulsations, gains
