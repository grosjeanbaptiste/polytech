"""L'enclenchement d'un circuit RL sur une tension sinusoidale.

Le syllabus d'electricite generale (16.20). A l'instant zero, on applique
`u = U racine(2) cos(omega t + theta_u)` a une resistance et une bobine
en serie. Le regime s'obtient par les substituts complexes,
`I = U / (R + j omega L)` (16.20.3) ; le phenomene transitoire le raccorde
au courant nul de depart :
`i = I racine(2) [cos(omega t + theta_u - phi) - exp(-t / tau) cos(theta_u - phi)]`
(16.20.8), avec `tau = L / R`.

L'importance du transitoire depend de l'instant ou l'on ferme
l'interrupteur : nul si `cos(theta_u - phi) = 0`, il est le plus grand
quand ce cosinus vaut un en valeur absolue, et le courant approche alors
le double de son amplitude de regime.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.circuits.enclenchement import enclenchement
from polytech.electromagnetisme.domain.symboles import TEMPS

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


def courant_a_l_enclenchement_rl(
    tension: Grandeur,
    phase: Grandeur,
    resistance: Grandeur,
    inductance: Grandeur,
    pulsation: Grandeur,
) -> sp.Expr:
    """`i(t)` de (16.20.8), le courant etant nul avant l'enclenchement."""
    reactance = pulsation * inductance
    efficace = tension / sp.sqrt(resistance**2 + reactance**2)
    ecart = sp.atan2(reactance, resistance)
    regime = efficace * sp.sqrt(2) * sp.cos(pulsation * TEMPS + phase - ecart)
    return enclenchement(regime, sp.Integer(0), inductance / sp.sympify(resistance))
