"""Macros de la feuille « Performances », exercices 4 a 7.

Prefixe `ExsHuit`.
"""

from __future__ import annotations

from polytech.signaux.application import black_exs_08 as bk
from polytech.signaux.application import courbes_exs_08 as co
from polytech.signaux.application import gains_exs_08 as ga
from polytech.signaux.application.formatage import nombre


def _lecture(prefixe: str, nom: str, duree: float) -> dict[str, str]:
    """Les caracteristiques de `M`, l'estimation et la simulation d'un exercice."""
    m, estimation = bk.caracteristiques(nom), bk.estimation(nom)
    finale, maximum, _ = bk.simulation(nom)
    rationnelle = bk.GAINS[nom] * bk.RATIONNELLES[nom]
    return {
        f"{prefixe}Continu": nombre(m.en_continu_en_decibels(), 2),
        f"{prefixe}Maximum": nombre(m.maximum_en_decibels(), 2),
        f"{prefixe}Bande": nombre(m.bande_passante, 3),
        f"{prefixe}Ecart": nombre(
            m.maximum_en_decibels() - m.en_continu_en_decibels(), 1
        ),
        f"{prefixe}Finale": nombre(estimation.finale, 3),
        f"{prefixe}Depassement": nombre(estimation.depassement, 1),
        f"{prefixe}MaximumEstime": nombre(estimation.maximum, 2),
        f"{prefixe}MonteeBasse": nombre(estimation.montee[0], 1),
        f"{prefixe}MonteeHaute": nombre(estimation.montee[1], 1),
        f"{prefixe}MaximumSimule": nombre(maximum, 3),
        f"{prefixe}DepassementSimule": nombre(100 * (maximum / finale - 1), 1),
        f"{prefixe}Indicielle": co.indicielle(rationnelle, bk.RETARDS[nom], duree),
    }


def _reglages_du_cinq() -> dict[str, str]:
    return {
        "ExsHuitCinqBlack": co.black_du_cinq(1.0),
        "ExsHuitCinqBlackDouble": co.black_du_cinq(2.0),
        "ExsHuitCinqPoints": co.points_du_tableau(),
        "ExsHuitCinqMargeDeGain": nombre(bk.marge_de_gain_en_decibels(1.0), 2),
        "ExsHuitCinqGain": nombre(bk.gain_pour_la_marge(90.0), 3),
    }


def _six_et_sept() -> dict[str, str]:
    six = ga.estimation_six()
    marge, limite = ga.marge_et_gain_limite()
    sept = ga.estimation_sept(0.5)
    return {
        "ExsHuitSixDepassement": nombre(six.depassement, 1),
        "ExsHuitSixMaximum": nombre(six.maximum, 3),
        "ExsHuitSixMonteeBasse": nombre(six.montee[0], 2),
        "ExsHuitSixMonteeHaute": nombre(six.montee[1], 2),
        "ExsHuitSeptMarge": nombre(marge, 2),
        "ExsHuitSeptLimite": nombre(limite, 3),
        "ExsHuitSeptDepassement": nombre(sept.depassement, 1),
        "ExsHuitSeptGain": nombre(ga.gain_pour_le_depassement(22.5), 2),
        "ExsHuitSeptIndicielle": co.indicielle(0.5 * ga.RATIONNELLE, ga.RETARD, 300.0),
    }


def macros_de_black() -> dict[str, str]:
    """Les macros des exercices 4 a 7."""
    return {
        **_lecture("ExsHuitQuatre", "quatre", 120.0),
        "ExsHuitQuatreModule": co.module_de_m("quatre", 0.01, 1.0),
        **_lecture("ExsHuitCinq", "cinq", 60.0),
        **_reglages_du_cinq(),
        **_six_et_sept(),
    }
