"""Macros du TP 5 d'electronique : gains, tensions, seuils de saturation.

Les tensions qui valent un rail (`+-12 V`) n'ont pas de macro : ce sont
des donnees de l'enonce, et le document les ecrit comme telles.

Le prefixe est `TpCinq`.
"""

from __future__ import annotations

from polytech.electronique.application import amplification_tp_05 as am
from polytech.electronique.application import combinaisons_tp_05 as cb
from polytech.electronique.application import montages_tp_05 as mt
from polytech.electronique.application import saturation_tp_05 as sa
from polytech.electronique.application.formatage import nombre


def _sans_aop() -> dict[str, str]:
    entree, sortie = am.amplificateur_reel()
    return {
        "TpCinqDiviseur": nombre(am.gain_du_diviseur(), 2),
        "TpCinqEntreeReelle": nombre(entree, 4),
        "TpCinqSortieReelle": nombre(sortie, 3),
        "TpCinqGainReel": nombre(sortie / 0.5, 2),
        "TpCinqDifferentiel": nombre(am.sortie_differentielle(), 1),
        "TpCinqBoucleOuverte": nombre(sa.boucle_ouverte()["a"], 2),
    }


def _un_aop() -> dict[str, str]:
    inverseur, non_inverseur = mt.gains_de_l_exercice_10()
    gain_13, courant_13 = mt.exercice_13()
    direct, _ = mt.exercice_14()
    simple, thevenin = mt.exercice_15()
    return {
        "TpCinqInverseur": nombre(inverseur, 1),
        "TpCinqNonInverseur": nombre(non_inverseur, 1),
        "TpCinqOnze": nombre(mt.gain_de_l_exercice_11(), 1),
        "TpCinqSeuilUn": nombre(sa.seuil_de_l_exercice_12(1e3), 4),
        "TpCinqSeuilVingt": nombre(sa.seuil_de_l_exercice_12(20e3), 2),
        "TpCinqCreteUn": nombre(sa.crete_de_l_exercice_12(1e3), 2),
        "TpCinqCreteVingt": nombre(sa.crete_de_l_exercice_12(20e3), 3),
        "TpCinqTreize": nombre(gain_13, 2),
        "TpCinqCourantTreize": nombre(courant_13 * 1e3, 2),
        "TpCinqDirect": nombre(direct, 2),
        "TpCinqQuinzeUn": nombre(simple, 2),
        "TpCinqQuinzeDeux": nombre(thevenin, 3),
    }


def _plusieurs_entrees() -> dict[str, str]:
    (a1, a2), (b1, b2) = cb.exercice_16()
    c1a, c2a = cb.coefficients_de_l_exercice_17(5e3)
    c1b, c2b = cb.coefficients_de_l_exercice_17(20e3)
    continu, alternatif = cb.exercice_19()
    return {
        "TpCinqSoustracteurUn": nombre(a1, 4),
        "TpCinqSoustracteurDeux": nombre(a2, 1),
        "TpCinqSoustracteur": nombre(a1 * 1 + a2 * 0.5, 4),
        "TpCinqSommateurUn": nombre(b1, 2),
        "TpCinqSommateurDeux": nombre(b2, 1),
        "TpCinqSommateur": nombre(b1 * 1 + b2 * 0.5, 1),
        "TpCinqDixSeptA": nombre(c1a, 1),
        "TpCinqDixSeptB": nombre(c2a, 1),
        "TpCinqDixSeptC": nombre(c1b, 1),
        "TpCinqDixSeptD": nombre(c2b, 1),
        "TpCinqDixHuit": nombre(cb.exercice_18(), 2),
        "TpCinqContinuMilli": nombre(continu * 1e3, 2),
        "TpCinqAlternatif": nombre(alternatif, 1),
    }


def _amplitudes() -> dict[str, str]:
    """Les sorties, gain fois l'amplitude d'entree que l'enonce donne."""
    inverseur, non_inverseur = mt.gains_de_l_exercice_10()
    gain_13, _ = mt.exercice_13()
    simple, thevenin = mt.exercice_15()
    c1a, c2a = cb.coefficients_de_l_exercice_17(5e3)
    c1b, c2b = cb.coefficients_de_l_exercice_17(20e3)
    _, alternatif = cb.exercice_19()
    return {
        "TpCinqDixInverseur": nombre(inverseur * 0.5, 1),
        "TpCinqDixNonInverseur": nombre(non_inverseur * 0.5, 2),
        "TpCinqTreizeSortie": nombre(gain_13 * 0.5, 2),
        "TpCinqQuinzeUnSortie": nombre(simple * 0.5, 3),
        "TpCinqQuinzeDeuxSortie": nombre(thevenin * 0.5, 4),
        "TpCinqDixSeptMille": nombre(c1a * 0.5, 2),
        "TpCinqDixSeptCinqCents": nombre(c2a * 0.05, 3),
        "TpCinqDixSeptSomme": nombre(c1b * 0.1 + c2b * 0.05, 2),
        "TpCinqAlternatifSortie": nombre(alternatif * 0.05, 2),
    }


def _intermediaires() -> dict[str, str]:
    """Les grandeurs que le raisonnement traverse, calculees elles aussi."""
    inverseur, _ = mt.gains_de_l_exercice_10()
    _, thevenin = mt.thevenin_de_l_exercice_15()
    return {
        "TpCinqSeuilMicro": nombre(sa.ALIMENTATION / 1e5 * 1e6, 0),
        "TpCinqEcartMinimal": nombre(-1 - (-3), 0),
        "TpCinqLineaireDouze": nombre(abs(inverseur) * 2, 0),
        "TpCinqViseeDouze": nombre(abs(sa.sortie_de_l_aop_12(1e3)) * 2, 0),
        "TpCinqTheveninKilo": nombre(thevenin / 1e3, 1),
        "TpCinqEntreeKilo": nombre((thevenin + 500 + 1e3) / 1e3, 0),
        "TpCinqDecalageMilli": nombre(cb.decalage_de_l_exercice_19() * 1e3, 0),
    }


def macros_du_tp_cinq() -> dict[str, str]:
    """Sans AOP, avec un AOP, avec plusieurs entrees ou etages."""
    return {
        **_sans_aop(),
        **_un_aop(),
        **_plusieurs_entrees(),
        **_amplitudes(),
        **_intermediaires(),
    }
