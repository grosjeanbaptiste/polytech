"""Les signaux usuels et leurs transformees de Laplace.

Syllabus de signaux et systemes, tableau 2.2. Ce fichier porte une
table : chaque ligne est une paire, l'original pour `t >= 0` et son
image. Les parametres `a`, `b`, `sigma`, `omega` et `alpha` sont reels,
et `rho^2 = sigma^2 + omega^2`.

Le denominateur `p^2 + 2 sigma p + rho^2` a deux poles complexes
conjugues `-sigma +/- j omega` tant que `sigma < rho` ; au-dela ils sont
reels, et le tableau l'ecrit `(p + a)(p + b)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import TEMPS as T

A, B, SIGMA, OMEGA, ALPHA = sp.symbols("a b sigma omega alpha", positive=True)
RHO: Final = sp.sqrt(SIGMA**2 + OMEGA**2)
SECOND_ORDRE: Final = P**2 + 2 * SIGMA * P + RHO**2
RAYON: Final = sp.sqrt(OMEGA**2 + (ALPHA - SIGMA) ** 2)
PHASE_DU_ZERO: Final = sp.atan((ALPHA - SIGMA) / OMEGA)
PART: Final = (ALPHA - A) / (B - A)
AMORTIE: Final = sp.exp(-SIGMA * T)


@dataclass(frozen=True, slots=True)
class Paire:
    """Un original, pour `t >= 0`, et son image."""

    original: sp.Expr
    image: sp.Expr


SIGNAUX_USUELS: Final = {
    "impulsion de Dirac": Paire(sp.DiracDelta(T), sp.S.One),
    "echelon": Paire(sp.S.One, 1 / P),
    "rampe": Paire(T, 1 / P**2),
    "exponentielle decroissante": Paire(sp.exp(-A * T), 1 / (P + A)),
    "exponentielle fois t": Paire(T * sp.exp(-A * T), 1 / (P + A) ** 2),
    "sinus": Paire(sp.sin(OMEGA * T) / OMEGA, 1 / (P**2 + OMEGA**2)),
    "cosinus": Paire(sp.cos(OMEGA * T), P / (P**2 + OMEGA**2)),
    "un moins cosinus": Paire(
        (1 - sp.cos(OMEGA * T)) / OMEGA**2, 1 / (P * (P**2 + OMEGA**2))
    ),
    "sinus amorti": Paire(AMORTIE * sp.sin(OMEGA * T) / OMEGA, 1 / SECOND_ORDRE),
    "reponse indicielle du second ordre": Paire(
        (1 - AMORTIE * RHO / OMEGA * sp.cos(OMEGA * T - sp.atan(SIGMA / OMEGA)))
        / RHO**2,
        1 / (P * SECOND_ORDRE),
    ),
    "cosinus amorti et dephase": Paire(
        RAYON / OMEGA * AMORTIE * sp.cos(OMEGA * T - PHASE_DU_ZERO),
        (P + ALPHA) / SECOND_ORDRE,
    ),
    "difference de deux exponentielles": Paire(
        (sp.exp(-A * T) - sp.exp(-B * T)) / (B - A), 1 / ((P + A) * (P + B))
    ),
    "melange de deux exponentielles": Paire(
        PART * sp.exp(-A * T) + (1 - PART) * sp.exp(-B * T),
        (P + ALPHA) / ((P + A) * (P + B)),
    ),
}
"""Le tableau 2.2, dans l'ordre du syllabus."""
