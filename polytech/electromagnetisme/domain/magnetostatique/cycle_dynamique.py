"""Le cycle d'hysteresis dynamique d'un barreau, et le flux qu'une tension impose.

Le syllabus d'electricite generale (17.7 et 17.8). Un barreau
ferromagnetique porte un enroulement de `N / l` spires par unite de
longueur. Le courant y determine le champ a la surface, `H = N i / l`
(17.7.1), et le flux l'induction moyenne `B = Phi / S` (17.7.2). Tracee
l'une contre l'autre a 50 Hz, ces deux grandeurs decrivent un cycle plus
large que le cycle statique : les courants de Foucault retardent le flux.

La puissance dissipee dans un troncon de volume `V` est `V H dB/dt`
(17.7.4), et l'energie d'un cycle `V` fois son aire (17.7.5). Les pertes
par hysteresis croissent comme la frequence, celles par courants de
Foucault comme son carre.

Si la resistance de l'enroulement est negligeable, une tension
sinusoidale impose un flux sinusoidal, `phi_max = U racine(2) / (N omega)`
(17.8.2) : c'est le champ, donc le courant, qui se deforme.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.symboles import TEMPS

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


def champ_a_la_surface(spires: float, courant: float, longueur: float) -> float:
    """`H = N i / l` (17.7.1), en amperes par metre."""
    if longueur <= 0:
        msg = f"longueur nulle ou negative : {longueur} m"
        raise ValueError(msg)
    return spires * courant / longueur


def induction_moyenne(flux: float, section: float) -> float:
    """`B = Phi / S` (17.7.2), en teslas."""
    if section <= 0:
        msg = f"section nulle ou negative : {section} m^2"
        raise ValueError(msg)
    return flux / section


def puissance_dissipee_dans_le_barreau(
    volume: Grandeur, champ: sp.Expr, induction: sp.Expr
) -> sp.Expr:
    """`p_t = V H dB/dt` (17.7.4), en watts."""
    return volume * champ * sp.diff(induction, TEMPS)


def pertes_magnetiques(hysteresis: float, foucault: float, frequence: float) -> float:
    """`k_h f + k_f f^2` : l'hysteresis en `f`, les courants de Foucault en `f^2`."""
    if frequence < 0:
        msg = f"frequence negative : {frequence} Hz"
        raise ValueError(msg)
    return hysteresis * frequence + foucault * frequence**2


def flux_impose(
    tension: Grandeur, spires: Grandeur, pulsation: Grandeur, phase: Grandeur
) -> sp.Expr:
    """`phi = (U racine(2) / (N omega)) sin(omega t + theta_u)` (17.8.2)."""
    crete = tension * sp.sqrt(2) / (spires * pulsation)
    return crete * sp.sin(pulsation * TEMPS + phase)
