"""La reponse impulsionnelle decomposee en modes, un par pole.

`H(p) = somme r_i / (p - p_i)` donne `h(t) = somme r_i e^{p_i t}`. Chaque
pole apporte un mode, et le **residu** `r_i` en fixe l'amplitude.

C'est ce qui separe deux notions que l'on confond volontiers. Un pole
**dominant** est le plus lent a s'eteindre ; il ne porte pas pour autant
le plus grand residu. Si les modes rapides ont de grands residus, ils
dessinent seuls le debut de la reponse, et le mode dominant n'en decrit
que la fin.

La decomposition exige une transmittance **strictement propre** : un
numerateur de meme degre que le denominateur ajouterait un terme direct,
c'est-a-dire une impulsion de Dirac dans `h(t)`, que nul mode ne decrit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import signal

if TYPE_CHECKING:
    from collections.abc import Sequence

TOLERANCE_DE_POLE = 1e-6
"""Un pole demande est reconnu a cette distance de l'un des poles calcules."""


def residus(
    numerateur: Sequence[float], denominateur: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Les residus et les poles ; leve sur une transmittance non strictement propre."""
    valeurs, poles, direct = signal.residue(numerateur, denominateur)
    if len(direct) > 0:
        msg = "transmittance non strictement propre : terme direct, donc un Dirac"
        raise ValueError(msg)
    return np.asarray(valeurs), np.asarray(poles)


def contribution_des_modes(
    numerateur: Sequence[float],
    denominateur: Sequence[float],
    retenus: np.ndarray,
    instants: np.ndarray,
) -> np.ndarray:
    """`somme r_i e^{p_i t}` sur les seuls poles retenus."""
    valeurs, poles = residus(numerateur, denominateur)
    total = np.zeros(len(instants), dtype=complex)
    for pole in retenus:
        rangs = np.flatnonzero(np.abs(poles - pole) < TOLERANCE_DE_POLE)
        if rangs.size == 0:
            msg = f"pole absent de la transmittance : {pole}"
            raise ValueError(msg)
        total += valeurs[rangs[0]] * np.exp(poles[rangs[0]] * instants)
    return np.asarray(total.real, dtype=float)
