"""Recueil de primitives, exercice 1 : les primitives 1 a 20.

Une ligne par primitive : le geste, et ce qu'il faut lui donner. Les
commentaires disent la reecriture ou le changement que le corrige montre.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.application.primitives_a_exs_01 import parties
from polytech.maths.application.recueil_primitives_enonces import (
    EXERCICE_UN as F,
)
from polytech.maths.application.recueil_primitives_enonces import A, B, N, P, R, X
from polytech.maths.application.recueil_primitives_gestes import (
    U,
    changement,
    immediate,
    terme_a_terme,
)
from polytech.maths.application.resolution_exs_01 import Resolution, par_equation

sin, cos, tan, exp, sqrt, ln = sp.sin, sp.cos, sp.tan, sp.exp, sp.sqrt, sp.log
ARC_MOITIE = "formule de l'arc moitie"
SEPARE = sp.Add(1 + tan(X) ** 2, -1, evaluate=False)
"""`tg^2 x = (1 + tg^2 x) - 1`, garde tel quel : `1 + tg^2` est la derivee de `tg`."""


def resolutions() -> dict[int, Resolution]:
    """Les vingt premieres primitives, chacune par son geste."""
    return {
        1: terme_a_terme(F[1], sp.expand(F[1]), "developpement"),
        2: parties(1 + 5 * X**2, exp(-X), gestes=2),
        3: changement(F[3], sqrt(U) / B, A + B * X),
        4: changement(F[4], (U - A) * sqrt(U) / B**2, A + B * X),
        5: changement(F[5], sqrt(R(7, 3)) / (7 * (1 + U**2)), sqrt(R(3, 7)) * X),
        6: changement(F[6], -1 / U, cos(X)),
        7: terme_a_terme(F[7], sqrt(2) * sin(X / 2), ARC_MOITIE),
        8: terme_a_terme(F[8], sqrt(2) * cos(X / 2), ARC_MOITIE),
        9: changement(F[9], exp(U) / (5 * ln(3)), 5 * X * ln(3)),
        10: changement(F[10], 1 / (1 + U**2), exp(X)),
        11: changement(F[11], -(U**N), cos(X)),
        12: terme_a_terme(F[12], SEPARE, "decomposition"),
        13: changement(F[13], U**18 * (U - 5) / 16, 4 * X + 5),
        14: changement(F[14], tan(U), ln(X)),
        15: immediate(F[15], -sp.cot(X)),
        16: par_equation((cos(P * X), exp(A * X)), (-P * sin(P * X) / A, exp(A * X))),
        17: changement(F[17], 1 / (10 * U), 5 * X**2 + 18),
        18: changement(F[18], U ** R(1, 5) / 10, 3 + 10 * exp(X)),
        19: par_equation((sin(P * X), exp(A * X)), (P * cos(P * X) / A, exp(A * X))),
        20: changement(F[20], -1 / (P * U), exp(-P * X) + B),
    }
