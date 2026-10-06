"""Macros du TP 7 d'electronique : biquads, echelles LC et filtres actifs.

Le prefixe est `TpSept`. Les cinq courbes de la figure portent une
lettre, de `A` (la moins amortie) a `E` : un nom de macro n'a pas de
chiffre.
"""

from __future__ import annotations

import math

from polytech.electronique.application import biquad_tp_07 as bq
from polytech.electronique.application import gain_tp_07 as ga
from polytech.electronique.application import sallen_key_tp_07 as sk
from polytech.electronique.application.formatage import coordonnees, nombre
from polytech.electronique.application.macros_echelles_tp_07 import (
    macros_des_echelles,
)
from polytech.electronique.domain import reponse_du_biquad as rb
from polytech.electronique.domain.methodes_sallen_key import Methode

_LETTRES = "ABCDE"
_RESONANTES = 3
"""Les trois premieres courbes ont un sommet : `xi < 1 / racine(2)`."""


def _biquad() -> dict[str, str]:
    macros = {
        "TpSeptCoupureLimite": nombre(rb.COUPURE_SANS_AMORTISSEMENT, 4),
        "TpSeptCoupureFigure": nombre(bq.coupure_de_la_figure(), 1),
        "TpSeptGainAuPoint": nombre(bq.gain_en_decibels(1552.387, 0.02), 4),
    }
    for rang, (lettre, xi) in enumerate(zip(_LETTRES, bq.AMORTISSEMENTS, strict=True)):
        qualite = 1 / (2 * xi)
        macros[f"TpSeptCourbe{lettre}"] = coordonnees(*bq.courbe(xi), 2)
        macros[f"TpSeptCoupureXi{lettre}"] = nombre(
            rb.coupure_a_trois_decibels(qualite), 3
        )
        if rang < _RESONANTES:
            sommet = 20 * math.log10(rb.surtension(qualite))
            macros[f"TpSeptSommetXi{lettre}"] = nombre(sommet, 2)
    return macros


def _sensibilite() -> dict[str, str]:
    cellule, a_dix = sk.cellule_du_4_1(), sk.cellule_du_4_1(resistance=10e3)
    a_un, a_cinq = (sk.pires_sensibilites_passives(q) for q in (1.0, 5.0))
    macros = {
        "TpSeptFacteurCapacites": nombre(sk.facteur_pour_les_capacites(), 3),
        "TpSeptFacteurCent": nombre(sk.facteur_pour_les_capacites(100e-6), 4),
        "TpSeptMinCUn": nombre(float(cellule.c1) * 1e9, 2),
        "TpSeptMinCDeux": nombre(float(cellule.c2) * 1e9, 3),
        "TpSeptDixCUn": nombre(float(a_dix.c1) * 1e12, 1),
        "TpSeptDixCDeux": nombre(float(a_dix.c2) * 1e12, 2),
        "TpSeptCoupureUne": nombre(sk.coupure_reelle_du_4_1() / 1e3, 1),
        "TpSeptCoupureTrois": nombre(sk.coupure_reelle_du_4_1(cellules=3) / 1e3, 1),
        "TpSeptBosseUne": nombre(sk.bosse_du_4_1(cellules=1), 2),
        "TpSeptBosseTrois": nombre(sk.bosse_du_4_1(cellules=3), 2),
        "TpSeptSensModeree": nombre(a_un[Methode.SENSIBILITE_MODEREE], 2),
        "TpSeptSensCinqA": nombre(a_cinq[Methode.ELEMENTS_EGAUX], 1),
        "TpSeptSensCinqB": nombre(a_cinq[Methode.CAPACITES_EGALES], 1),
        "TpSeptSensCinqC": nombre(a_cinq[Methode.SENSIBILITE_MODEREE], 2),
    }
    for lettre, six in zip("ABC", sk.cellules_de_butterworth(), strict=True):
        macros[f"TpSeptSixQ{lettre}"] = nombre(float(six.qualite()), 3)
        macros[f"TpSeptSixCUn{lettre}"] = nombre(float(six.c1) * 1e9, 2)
        macros[f"TpSeptSixCDeux{lettre}"] = nombre(float(six.c2) * 1e9, 2)
    return macros


def _gain() -> dict[str, str]:
    cellule = ga.cellule()
    vers_la_masse, retour = ga.pont_compense()
    officiel = ga.pont_officiel()
    frequence, au_sommet = ga.sommet()
    return {
        "TpSeptResistanceNormalisee": nombre(ga.resistance_normalisee() * 1e6, 2),
        "TpSeptFacteurGain": nombre(max(ga.facteurs_d_echelle()) / 1e7, 3),
        "TpSeptGainC": nombre(float(cellule.c1) * 1e9, 2),
        "TpSeptPontA": nombre(vers_la_masse / 1e3, 2),
        "TpSeptPontB": nombre(retour / 1e3, 2),
        "TpSeptPontOfficiel": nombre(officiel[0] * officiel[1] / sum(officiel), 0),
        "TpSeptBandePlate": nombre(ga.bande_plate() / 1e3, 2),
        "TpSeptSommetFrequence": nombre(frequence / 1e3, 2),
        "TpSeptSommetGain": nombre(au_sommet, 2),
        "TpSeptCreteSommet": nombre(ga.crete_au_sommet(), 1),
        "TpSeptEntreeMaximale": nombre(ga.ALIMENTATION / au_sommet, 2),
        "TpSeptCoupureGain": nombre(ga.coupure_reelle() / 1e3, 2),
        "TpSeptPropre": nombre(
            ga.frequence_propre_pour_couper_a_dix_kilohertz() / 1e3, 2
        ),
    }


def macros_du_tp_sept() -> dict[str, str]:
    """Toutes les valeurs que le corrige du TP 7 compose."""
    return {**_biquad(), **macros_des_echelles(), **_sensibilite(), **_gain()}
