"""Le calcul d'un PID par l'approche frequentielle.

Planches « Controleur » du cours de signaux et systemes. Le controleur
deplace la courbe de Black du procede : en chaque pulsation il ajoute
son gain en decibels et son argument. Avec trois parametres, on ne peut
imposer le deplacement que d'un point. On choisit donc `alpha` (dix, le
plus souvent), un point cible `P` dans la zone permise -- d'abord
`(-120 degres, 0 dB)` --, et une pulsation de correction `w_c` : le
point de la courbe du procede en `w_c` doit arriver en `P`.

Deux equations en decoulent. Celle des arguments donne `T_i`, puis
`T_d = T_i / alpha` ; celle des modules donne `G_c`. L'argument du PID
etant borne, toutes les pulsations ne conviennent pas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy.optimize import brentq

from polytech.signaux.domain.marges import arguments_deroules
from polytech.signaux.domain.pid_serie import REDUITES, PidSerie
from polytech.signaux.domain.symboles import IMAGE as P

ARGUMENT_MINIMAL = -90
"""En degres : l'argument du PID aux tres basses frequences."""


@dataclass(frozen=True, slots=True)
class Cible:
    """Le point vise dans le plan de Black : argument en degres, gain en dB."""

    argument: float
    gain: float


def _au_point_de_depart(procede: sp.Expr, pulsation: float) -> tuple[float, float]:
    """Le module du procede en `w_c`, et son argument deroule, en degres."""
    pulsations = np.logspace(-4, math.log10(pulsation), 20_001)
    valeurs = np.asarray(sp.lambdify(P, procede, "numpy")(1j * pulsations))
    phase = arguments_deroules(procede, valeurs.astype(complex))[-1]
    return (float(abs(valeurs[-1])), math.degrees(phase))


def synthetiser_le_pid(
    procede: sp.Expr,
    cible: Cible,
    *,
    pulsation: float,
    alpha: float,
    filtre: float = 0.1,
) -> PidSerie:
    """Le PID qui amene en `cible` le point de pulsation `w_c` du procede."""
    module, argument = _au_point_de_depart(procede, pulsation)
    unitaire = PidSerie(sp.S.One, sp.Float(alpha), sp.S.One, sp.Float(filtre))
    avance = cible.argument - argument
    arguments = [unitaire.argument(float(x)) for x in REDUITES[::40]]
    sommet = int(np.argmax(arguments))
    if not ARGUMENT_MINIMAL < avance < arguments[sommet]:
        msg = f"point cible hors de portee du PID : il faudrait {avance:.1f} degres"
        raise ValueError(msg)
    borne = float(REDUITES[::40][sommet])
    reduite = brentq(lambda x: unitaire.argument(x) - avance, 1e-6, borne)
    integrale = reduite / pulsation
    sans_gain = PidSerie(sp.S.One, sp.Float(integrale), sp.Float(integrale / alpha))
    apport = abs(complex(sans_gain.transmittance().subs(P, 1j * pulsation)))
    gain = 10 ** (cible.gain / 20) / (module * apport)
    return PidSerie(sp.Float(gain), sans_gain.integrale, sans_gain.derivee)
