"""Macros de la seance 3.2, exercice 6 : deux circuits couples.

Les courants viennent du resolveur, couplage compris ; le rapport
`I_1 / I_2` se lit aussi sur la seule maille secondaire, et c'est ce que
fait le manuscrit. Les deux chemins sont publies : les reactances qui
entrent dans le rapport, puis le rapport lui-meme.

La puissance de la branche `R_2`-`C_2` ne depend que du module de `I_2`,
et non de son sens : le signe impose par les points de couplage ne peut
donc pas la fausser. Le dephasage, lui, en depend -- c'est pourquoi la
figure est lue avant de l'ecrire.

Le prefixe est `TroisDeuxSix`.
"""

from __future__ import annotations

import cmath
import math

import sympy as sp

from polytech.electromagnetisme.application import symboles_seance_32 as d
from polytech.electromagnetisme.application.enonces_seance_32 import (
    PRIMAIRE_DU_SIX,
    circuit_du_six,
    courant_secondaire_du_six,
)
from polytech.electromagnetisme.application.formatage import nombre
from polytech.electromagnetisme.domain.circuits.phaseurs import (
    en_regime_sinusoidal,
    puissance_complexe,
)


def _numerique(expression: sp.Expr) -> complex:
    """La valeur d'un phaseur exact, a la pulsation de l'exercice."""
    return complex(sp.N(en_regime_sinusoidal(expression, d.OMEGA_DU_SIX), 30))


def _reactances() -> dict[str, str]:
    """Les trois reactances et la maille secondaire."""
    omega = float(d.OMEGA_DU_SIX)
    capacitive = 1 / (omega * float(d.SIX_C2))
    secondaire = complex(float(d.SIX_R2), omega * float(d.SIX_L2) - capacitive)
    rapport = secondaire / complex(0, omega * float(d.SIX_L12))
    return {
        "TroisDeuxSixXL": nombre(omega * float(d.SIX_L1), 2),
        "TroisDeuxSixXLDeux": nombre(omega * float(d.SIX_L2), 2),
        "TroisDeuxSixXC": nombre(capacitive, 2),
        "TroisDeuxSixXM": nombre(omega * float(d.SIX_L12), 2),
        "TroisDeuxSixZDeuxImag": nombre(-secondaire.imag, 2),
        "TroisDeuxSixRapport": nombre(abs(rapport), 4),
        "TroisDeuxSixRapportReel": nombre(-rapport.real, 4),
        "TroisDeuxSixRapportImag": nombre(-rapport.imag, 4),
    }


def macros_du_six() -> dict[str, str]:
    """Les donnees, les reactances, le dephasage, la puissance."""
    reseau = circuit_du_six()
    primaire = _numerique(reseau.courant(PRIMAIRE_DU_SIX))
    secondaire = _numerique(courant_secondaire_du_six())
    impedance = complex(float(d.SIX_R2), -1 / (float(d.OMEGA_DU_SIX * d.SIX_C2)))
    puissance = complex(
        sp.N(puissance_complexe(impedance * secondaire, secondaire), 30),
    )
    return {
        "TroisDeuxSixFrequence": nombre(float(d.SIX_FREQUENCE), 0),
        "TroisDeuxSixE": nombre(float(d.SIX_E1), 0),
        "TroisDeuxSixRUn": nombre(float(d.SIX_R1), 0),
        "TroisDeuxSixRDeux": nombre(float(d.SIX_R2), 0),
        "TroisDeuxSixLUn": nombre(float(d.SIX_L1) * 1e3, 0),
        "TroisDeuxSixLDeux": nombre(float(d.SIX_L2) * 1e3, 0),
        "TroisDeuxSixLM": nombre(float(d.SIX_L12) * 1e3, 0),
        "TroisDeuxSixCDeux": nombre(float(d.SIX_C2) * 1e6, 0),
        **_reactances(),
        "TroisDeuxSixDephasage": nombre(
            math.degrees(cmath.phase(primaire / secondaire)), 3
        ),
        "TroisDeuxSixCourantDeux": nombre(abs(secondaire) * 1e3, 2),
        "TroisDeuxSixCourantUn": nombre(abs(primaire) * 1e3, 2),
        "TroisDeuxSixP": nombre(puissance.real * 1e3, 3),
        "TroisDeuxSixQ": nombre(puissance.imag, 4),
        "TroisDeuxSixCosPhi": nombre(math.cos(cmath.phase(puissance)), 4),
    }
