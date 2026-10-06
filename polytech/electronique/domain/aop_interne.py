"""Ce que la structure interne d'un amplificateur operationnel impose.

Planches du chapitre 2, partie II. Un amplificateur a trois etages -- une
paire differentielle, un etage a grand gain, un etage de puissance -- et
chacun laisse sa marque.

* Une capacite de retour autour d'un gain `K` se ramene, par le theoreme
  de **Miller**, a deux impedances vers la masse : `Z / (1 - K)` a
  l'entree et `Z / (1 - 1/K)` a la sortie. Avec `K = -A`, la capacite
  parait `(1 + A)` fois plus grande : c'est ce qui compense l'etage a
  grand gain, et ce qui en limite la bande.
* Le gain n'a alors qu'un pole, `A_v(s) = 2 pi GBW / (s + 2 pi BW_ol)`,
  et la retroaction conserve le produit gain-bande.
* La paire differentielle ne fournit qu'un courant borne, `I_x = I_1
  tanh(V_id / 2 v_T)`. Sature, il charge la capacite de compensation a
  pente constante : la **vitesse de balayage** `I_1 / C_f`.
* Les deux entrees tirent un courant, leur moyenne est le courant de
  polarisation, leur difference le courant de decalage ; le mode commun
  et les alimentations passent en sortie, divises par leur taux de
  rejection.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

TENSION_THERMIQUE: Final = 25e-3
"""`v_T`, en volts : « ~ 25 mV at T = 25 C » (planche 34)."""


def impedances_de_miller(impedance: sp.Expr, gain: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`(Z / (1 - K), Z / (1 - 1/K))` : l'entree, puis la sortie."""
    if sp.simplify(gain - 1) == 0:
        msg = "gain unite : aucune tension aux bornes de l'impedance de retour"
        raise ValueError(msg)
    return impedance / (1 - gain), impedance / (1 - 1 / gain)


def bande_en_boucle_ouverte(produit: float, continu: float) -> float:
    """`BW_ol = GBW / A_ol(0)`, en hertz."""
    if continu <= 0:
        msg = f"gain en continu nul ou negatif : {continu}"
        raise ValueError(msg)
    return produit / continu


def gain_ferme_a_un_pole(
    variable: sp.Symbol, *, produit: float, continu: float, retour: float
) -> sp.Expr:
    """`A_cl(s) = A_ol(s) / (1 + beta A_ol(s))`, pour un gain ouvert a un pole."""
    pulsation = 2 * sp.pi * produit
    ouvert = pulsation / (variable + pulsation / continu)
    return sp.simplify(ouvert / (1 + retour * ouvert))


def courant_de_la_paire_differentielle(
    differentielle: float, queue: float, *, thermique: float = TENSION_THERMIQUE
) -> float:
    """`I_x = I_1 tanh(V_id / (2 v_T))`, borne par le courant de queue `I_1`."""
    return queue * math.tanh(differentielle / (2 * thermique))


def courants_de_la_paire(
    differentielle: float, queue: float, *, thermique: float = TENSION_THERMIQUE
) -> tuple[float, float]:
    """`(I_1 / (1 + e^-x), I_1 / (1 + e^x))`, `x = V_id / v_T` : les deux branches."""
    reduit = differentielle / thermique
    return queue / (1 + math.exp(-reduit)), queue / (1 + math.exp(reduit))


def module_du_gain_ouvert(frequence: float, *, produit: float, continu: float) -> float:
    """`|GBW / (j f + GBW / A_ol(0))|` : le gain a un pole, a cette frequence."""
    if frequence < 0:
        msg = f"frequence negative : {frequence} Hz"
        raise ValueError(msg)
    return produit / math.hypot(frequence, bande_en_boucle_ouverte(produit, continu))


def vitesse_de_balayage(queue: float, compensation: float) -> float:
    """`SR = I_1 / C_f`, en V/s : la pente que la sortie ne depasse pas."""
    if compensation <= 0:
        msg = f"capacite de compensation nulle ou negative : {compensation} F"
        raise ValueError(msg)
    return queue / compensation


def courant_de_polarisation(directe: float, inverseuse: float) -> float:
    """`I_BIAS = (I_B+ + I_B-) / 2`."""
    return (directe + inverseuse) / 2


def courant_de_decalage(directe: float, inverseuse: float) -> float:
    """`I_OS = I_B+ - I_B-`."""
    return directe - inverseuse


def tension_de_mode_commun(directe: float, inverseuse: float) -> float:
    """`V_cm = (V+ + V-) / 2`."""
    return (directe + inverseuse) / 2


def sortie_avec_imperfections(
    gain: float,
    differentielle: float,
    *,
    mode_commun: tuple[float, float] | None = None,
    alimentations: tuple[tuple[float, float], ...] = (),
) -> float:
    """`A_v (V_id + V_cm / CMRR + somme(dV / PSRR))`, hors saturation.

    `mode_commun` est `(V_cm, CMRR)` ; chaque alimentation, `(dV, PSRR)`.
    """
    ramenee = differentielle
    parasites = (*((mode_commun,) if mode_commun else ()), *alimentations)
    for tension, rejection in parasites:
        if rejection <= 0:
            msg = f"taux de rejection nul ou negatif : {rejection}"
            raise ValueError(msg)
        ramenee += tension / rejection
    return gain * ramenee
