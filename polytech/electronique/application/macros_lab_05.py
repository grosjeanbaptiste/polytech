"""Macros du laboratoire 5 d'electronique : l'amplificateur audio.

Le prefixe est `LabCinq`.
"""

from __future__ import annotations

from polytech.electronique.application import chaine_lab_05 as ch
from polytech.electronique.application import decalage_lab_05 as dc
from polytech.electronique.application import dimensionnement_lab_05 as dm
from polytech.electronique.application import egaliseur_lab_05 as eg
from polytech.electronique.application import sortie_lab_05 as so
from polytech.electronique.application.formatage import coordonnees, nombre


def _etages() -> dict[str, str]:
    premier, second = dm.gains_realises()
    bande_un, bande_deux = dm.bandes_des_etages()
    avec, sans = dc.decalages(), dc.decalages(courants=False)
    officiel = dc.decalages(vers_la_masse=10e3, contre_reaction=100e3)
    return {
        "LabCinqGainSecond": nombre(second, 2),
        "LabCinqGainDeuxEtages": nombre(premier * second, 1),
        "LabCinqBandeUn": nombre(bande_un / 1e3, 0),
        "LabCinqBandeDeux": nombre(bande_deux / 1e3, 0),
        "LabCinqDecalageDeux": nombre(avec[1] * 1e3, 1),
        "LabCinqCourantContinu": nombre(abs(avec[1]) / dm.HAUT_PARLEUR * 1e3, 2),
        "LabCinqOfficielUn": nombre(officiel[0] * 1e3, 1),
        "LabCinqOfficielDeux": nombre(officiel[1] * 1e3, 1),
        "LabCinqCourantUn": nombre(abs(avec[0] - sans[0]) * 1e9, 1),
        "LabCinqCourantDeux": nombre(abs(avec[1] - sans[1]) * 1e9, 0),
    }


def _sortie() -> dict[str, str]:
    return {
        "LabCinqPulsation": nombre(so.pulsation_de_coupure(), 1),
        "LabCinqCoupure": nombre(so.frequence_de_coupure(), 1),
        "LabCinqLiaison": nombre(so.condensateur_de_liaison() * 1e6, 0),
        "LabCinqPuissancePont": nombre(so.puissance_maximale_sur_la_charge(), 2),
    }


def _egaliseur() -> dict[str, str]:
    absolue, relative = eg.coupures_des_basses()
    frequences, basses, aigus, somme = eg.courbes()
    return {
        "LabCinqSeparation": nombre(eg.frequence_de_separation(), 1),
        "LabCinqCoupureAbsolue": nombre(absolue, 1),
        "LabCinqCoupureRelative": nombre(relative, 1),
        "LabCinqPlateau": nombre(eg.plateau_des_basses(), 3),
        "LabCinqOndulation": nombre(eg.ondulation_de_la_somme(), 2),
        "LabCinqGainBasses": nombre(eg.gain_des_basses(100.0), 2),
        "LabCinqGainAigus": nombre(eg.gain_des_aigus(1000.0), 2),
        "LabCinqCourbeBasses": coordonnees(frequences, basses, 2),
        "LabCinqCourbeAigus": coordonnees(frequences, aigus, 2),
        "LabCinqCourbeSomme": coordonnees(frequences, somme, 2),
    }


def _chaine() -> dict[str, str]:
    basse, haute = ch.bande_a_trois_decibels()
    return {
        "LabCinqGainMaximal": nombre(ch.gain_maximal(), 1),
        "LabCinqCreteDemandee": nombre(ch.crete_demandee(), 2),
        "LabCinqReglage": nombre(ch.potentiometre_pour_le_gain(160.0) / 1e3, 0),
        "LabCinqBandeSommateur": nombre(ch.bande_du_sommateur() / 1e3, 0),
        "LabCinqPerteBasse": nombre(ch.perte(20.0), 2),
        "LabCinqPerteHaute": nombre(ch.perte(20e3), 2),
        "LabCinqBandeBasse": nombre(basse, 1),
        "LabCinqBandeHaute": nombre(haute / 1e3, 0),
    }


def macros_du_labo_cinq() -> dict[str, str]:
    """Toutes les valeurs que le corrige du laboratoire 5 compose."""
    return {**_etages(), **_sortie(), **_egaliseur(), **_chaine()}
