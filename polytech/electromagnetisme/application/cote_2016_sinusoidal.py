"""Exercice cote 2016, questions 4 et 5 : regime sinusoidal, puis une ouverture."""

from __future__ import annotations

import cmath
import math

import numpy as np

from polytech.electromagnetisme.application import cote_2016_enonce as en


def _impedances() -> tuple[complex, complex, complex]:
    """Les trois branches : `R1`, `R2` en serie avec `C2`, `R3` avec `L3`."""
    mediane = en.R2_Q4 + 1 / (1j * en.PULSATION * en.C2)
    return complex(en.R1_Q4), mediane, en.R3 + 1j * en.PULSATION * en.L3


def courants_de_mailles() -> tuple[complex, complex]:
    """`I1` et `I2`, solutions des deux mailles du manuscrit."""
    z1, mediane, z3 = _impedances()
    e1 = complex(en.TENSION_EFFICACE)
    e2 = cmath.rect(en.TENSION_EFFICACE, math.radians(en.AVANCE_DE_E2_EN_DEGRES))
    systeme = np.array([[z1 + mediane, mediane], [mediane, mediane + z3]])
    i1, i2 = np.linalg.solve(systeme, np.array([e1 - e2, -e2]))
    return complex(i1), complex(i2)


def courant_de_la_branche() -> complex:
    """Le courant de la branche `L3`-`R3` : `-I2`, avec le `I2` de l'enonce."""
    return -en.I2_DONNE


def tension_de_la_branche() -> complex:
    """`U = (R3 + j omega L3) I`."""
    return _impedances()[2] * courant_de_la_branche()


def puissance_de_la_branche() -> complex:
    """`S = U I*`, phaseurs efficaces."""
    return tension_de_la_branche() * courant_de_la_branche().conjugate()


def puissances_par_le_courant() -> tuple[float, float]:
    """`P = R3 |I|^2` et `Q = omega L3 |I|^2`, sans passer par la tension."""
    carre = abs(courant_de_la_branche()) ** 2
    return en.R3 * carre, en.PULSATION * en.L3 * carre


def dephasage_du_courant() -> float:
    """`theta_I - theta_U`, en degres : negatif, le courant retarde."""
    ecart = cmath.phase(courant_de_la_branche()) - cmath.phase(tension_de_la_branche())
    return math.degrees(ecart)


def tension_a_l_ouverture(phase: float = en.PHASE) -> float:
    """`u_C(0) = e(0)` : ferme, l'interrupteur met `C` aux bornes de la source."""
    return en.E0 * math.cos(phase)


def courant_apres_ouverture(instant: float, phase: float = en.PHASE) -> float:
    """`i(t) = u_C(0)/R e^(-t/RC)` : `C` se decharge dans `R`, seul."""
    depart = tension_a_l_ouverture(phase) / en.R
    return depart * math.exp(-instant / (en.R * en.C))
