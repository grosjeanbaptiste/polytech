"""Macros du TP 6 d'electronique : etages, decalages, polarisation, choix d'AOP.

Le prefixe est `TpSix`.
"""

from __future__ import annotations

import math

from polytech.electronique.application import decalage_tp_06 as dc
from polytech.electronique.application import dimensionnement_tp_06 as dm
from polytech.electronique.application import polarisation_tp_06 as po
from polytech.electronique.application.formatage import nombre


def _etages_et_decalages() -> dict[str, str]:
    e0, plus, moins = dc.decalage_lm741_par_source()
    i_plus, i_moins = dc.courants_d_entree(80e-9, 20e-9)
    ca_plus, ca_moins = dc.courants_d_entree(10e-12, 0.5e-12)
    return {
        "TpSixBandeEtage": nombre(dm.bande_d_un_etage(15), 0),
        "TpSixEtagesCours": str(dm.etages_selon_le_cours()),
        "TpSixEtagesChaine": str(dm.etages_selon_la_chaine()),
        "TpSixBandeDeux": nombre(dm.bande_de_la_chaine(2) / 1e3, 1),
        "TpSixBandeTrois": nombre(dm.bande_de_la_chaine(3) / 1e3, 1),
        "TpSixIPlus": nombre(i_plus * 1e9, 0),
        "TpSixIMoins": nombre(i_moins * 1e9, 0),
        "TpSixPartEzero": nombre(e0 * 1e3, 0),
        "TpSixPartPlus": nombre(plus * 1e3, 0),
        "TpSixPartMoins": nombre(moins * 1e3, 2),
        "TpSixDecalageLm": nombre(dc.decalage_lm741() * 1e3, 2),
        "TpSixCaPlus": nombre(ca_plus * 1e12, 2),
        "TpSixCaMoins": nombre(ca_moins * 1e12, 2),
        "TpSixDecalageCa": nombre(dc.decalage_ca3140a() * 1e3, 3),
        "TpSixDecalageSept": nombre(dc.decalage_de_l_exercice_7() * 1e3, 1),
    }


def _polarisation_et_choix() -> dict[str, str]:
    module, phase = dm.suiveur_opa549_a_deux_megahertz()
    gains = dm.gains_maximaux_a_deux_megahertz()
    a_vingt = po.module_a_vingt_hertz()
    return {
        "TpSixRepos": nombre(po.repos_de_sortie(), 1),
        "TpSixGainAlternatif": nombre(po.gain_alternatif(), 0),
        "TpSixModuleVingt": nombre(a_vingt, 1),
        "TpSixPerteVingt": nombre(20 * math.log10(a_vingt / 100), 2),
        "TpSixCapaciteMinimale": nombre(1 / (2 * math.pi * 20 * 1e3) * 1e6, 2),
        "TpSixCoupureSortie": nombre(po.coupure_du_filtre_de_sortie(), 2),
        "TpSixGainQuatorze": nombre(10 ** (14 / 20), 3),
        "TpSixBandeSept": nombre(dm.bande_de_l_inverseur(10e3, 50e3, 1.5e6) / 1e3, 0),
        "TpSixBandeSeptCours": nombre(dm.bande_d_un_etage(5) / 1e3, 0),
        "TpSixBandeHuit": nombre(dm.bande_d_un_etage(10), 0),
        "TpSixPenteHuit": nombre(2 * math.pi * dm.BANDE_AUDIO / 1e6, 3),
        "TpSixGainAd": f"{gains['AD8129']:.1f}",
        "TpSixGainOpaCinq": nombre(gains["OPA549"], 2),
        "TpSixGainOpaUn": f"{gains['OPA132']:.1f}",
        "TpSixPenteNeuf": nombre(dm.pente_requise(), 1),
        "TpSixModuleSuiveur": nombre(module, 3),
        "TpSixPhaseSuiveur": nombre(phase, 1),
        "TpSixSurAmplification": nombre(1 / gains["OPA549"], 1),
    }


def macros_du_tp_six() -> dict[str, str]:
    """Les etages et les decalages, puis la polarisation et les choix d'AOP."""
    return {**_etages_et_decalages(), **_polarisation_et_choix()}
