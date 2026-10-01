"""Les defauts de l'AOP reel, et ce qu'ils coutent au montage.

Le fascicule en traite cinq, et chacun limite une chose differente.

Le **taux de rejection du mode commun** dit ce qui passe d'une tension
appliquee aux deux entrees a la fois. La **plage d'entree en mode commun**
dit jusqu'ou cette tension peut aller sans que l'etage d'entree sorte de
son domaine -- et elle n'atteint jamais les alimentations.

Le **courant de polarisation** cree une tension d'erreur sur les
resistances du montage ; il se compense en egalisant les impedances vues
par les deux entrees. Il ne reste alors que le courant de **decalage**,
d'un ordre de grandeur plus faible -- c'est tout l'interet de la
compensation. La **tension de decalage**, elle, subit le gain du montage :
un millivolt devient cent millivolts a gain cent.

Et la **capacite de Miller** surprend : la capacite vue a l'entree d'un
inverseur de gain `-G` vaut `C (1 + G)`. Un gain de cent multiplie par
cent une capacite parasite de quelques picofarads, et c'est elle qui
limite la bande du montage bien avant l'AOP.
"""

from __future__ import annotations

import math

DECIBELS_DE_TENSION = 20
"""Le taux de rejection est un rapport de tensions : 20 log, pas 10."""


def rejection_du_mode_commun(
    gain_differentiel: float,
    gain_de_mode_commun: float,
) -> float:
    """`CMRR = |A_d / A_c|` ; leve si le gain de mode commun est nul."""
    if gain_de_mode_commun == 0:
        msg = (
            "gain de mode commun nul : le taux de rejection serait infini, "
            "ce qu'aucun composant reel n'atteint"
        )
        raise ValueError(msg)
    return abs(gain_differentiel / gain_de_mode_commun)


def rejection_en_decibels(taux: float) -> float:
    """`20 log10(CMRR)`, en decibels."""
    if taux <= 0:
        msg = f"taux de rejection nul ou negatif : {taux}"
        raise ValueError(msg)
    return DECIBELS_DE_TENSION * math.log10(taux)


def plage_d_entree_en_mode_commun(
    alimentation_basse: float,
    alimentation_haute: float,
    marge: float,
) -> tuple[float, float]:
    """Les deux bornes de la plage, retraites des alimentations par la marge."""
    if alimentation_haute <= alimentation_basse:
        msg = (
            f"alimentations inversees : {alimentation_haute} <= {alimentation_basse} V"
        )
        raise ValueError(msg)
    if marge < 0 or 2 * marge >= alimentation_haute - alimentation_basse:
        msg = (
            f"marge de {marge} V incompatible avec une alimentation de "
            f"{alimentation_haute - alimentation_basse} V"
        )
        raise ValueError(msg)
    return alimentation_basse + marge, alimentation_haute - marge


def erreur_de_courant_de_polarisation(courant: float, resistance: float) -> float:
    """`V = I_b R`, en volts : l'erreur avant toute compensation."""
    if resistance < 0:
        msg = f"resistance negative : {resistance} ohm"
        raise ValueError(msg)
    return courant * resistance


def resistance_de_compensation(vers_la_masse: float, contre_reaction: float) -> float:
    """`R_1 // R_f` : l'impedance a placer sur l'autre entree.

    Elle egalise ce que les deux entrees voient, donc annule l'effet du
    courant de polarisation -- qui est **commun** aux deux. Ne reste que
    le courant de decalage, qui est leur difference.
    """
    if vers_la_masse <= 0 or contre_reaction <= 0:
        msg = f"resistance nulle ou negative : {vers_la_masse} et {contre_reaction} ohm"
        raise ValueError(msg)
    return vers_la_masse * contre_reaction / (vers_la_masse + contre_reaction)


def erreur_de_courant_de_decalage(courant: float, resistance: float) -> float:
    """`V = I_os R` : ce qui subsiste apres compensation."""
    if resistance < 0:
        msg = f"resistance negative : {resistance} ohm"
        raise ValueError(msg)
    return courant * resistance


def erreur_de_tension_de_decalage(decalage: float, gain: float) -> float:
    """`V_os G` : la tension de decalage subit le gain du montage."""
    if gain < 0:
        msg = f"gain negatif : {gain} -- employer son module"
        raise ValueError(msg)
    return decalage * gain


def capacite_de_miller(capacite: float, module_du_gain: float) -> float:
    """`C (1 + G)` : la capacite vue a l'entree d'un inverseur de gain `-G`."""
    if module_du_gain < 0:
        msg = f"module du gain negatif : {module_du_gain}"
        raise ValueError(msg)
    if capacite < 0:
        msg = f"capacite negative : {capacite} F"
        raise ValueError(msg)
    return capacite * (1 + module_du_gain)
