"""Macros du laboratoire 3 d'electronique : quatre chaines d'acquisition.

Le prefixe est `LabTrois`, suivi de la lettre de l'exercice : `A` pour
1.1, `B` pour 1.2, `C` pour 1.3, `D` pour 1.4. Les puissances de bruit
sont publiees en microvolts carres, sauf celles de l'exercice 1.1, que
le corrige officiel donne en sortie de chaine : en millivolts carres.
"""

from __future__ import annotations

from polytech.electronique.application import budget_1_2_lab_03 as budget
from polytech.electronique.application import chaine_1_1_lab_03 as un
from polytech.electronique.application import chaine_1_2_lab_03 as deux
from polytech.electronique.application import chaine_1_3_lab_03 as trois
from polytech.electronique.application import chaine_1_4_lab_03 as quatre
from polytech.electronique.application.donnees_lab_03 import UnDeux
from polytech.electronique.application.formatage import nombre

_MICRO_CARRE = 1e12
_MILLI_CARRE = 1e6


def _premier() -> dict[str, str]:
    a, haut, b, bas, can = (p * _MILLI_CARRE for p in un.puissances().values())
    total, efficace, densite = un.bruit_total()
    return {
        "LabTroisAGainExact": nombre(un.gain_exact_du_premier_etage(), 3),
        "LabTroisASortieMax": nombre(un.sortie_maximale_du_premier_etage(), 3),
        "LabTroisAPas": nombre(un.pas_a_l_entree(un.bits_minimaux()) * 1e6, 3),
        "LabTroisAAmpliA": nombre(a, 1),
        "LabTroisAHaut": nombre(haut, 1),
        "LabTroisAAmpliB": nombre(b, 2),
        "LabTroisABas": nombre(bas, 4),
        "LabTroisACan": nombre(can, 2),
        "LabTroisATotal": nombre(total * _MILLI_CARRE, 4),
        "LabTroisAEfficace": nombre(efficace * 1e3, 3),
        "LabTroisADensite": nombre(densite * 1e6, 2),
        "LabTroisAPart": nombre(100 * un.etage_dominant()[1], 0),
        "LabTroisAEntree": nombre(un.bruit_a_l_entree() * 1e6, 2),
        "LabTroisAGainTotal": nombre(un.gains()[0], 0),
        "LabTroisAGainA": nombre(un.gains()[1], 0),
        "LabTroisAGainB": nombre(un.gains()[2], 0),
    }


def _deuxieme() -> dict[str, str]:
    chaine = {n: p * _MICRO_CARRE for n, p in budget.puissances_de_la_chaine().items()}
    return {
        "LabTroisBCadence": nombre(deux.cadence_totale() / 1e3, 0),
        "LabTroisBMUn": nombre(deux.bruit_du_multiplexeur("M1") * _MICRO_CARRE, 2),
        "LabTroisBMDeux": nombre(deux.bruit_du_multiplexeur("M2") * _MICRO_CARRE, 2),
        "LabTroisBCourant": nombre(
            deux.pire_bruit_en_courant() * _MICRO_CARRE * 1e7, 3
        ),
        "LabTroisBReste": nombre(deux.budget_restant() * _MICRO_CARRE, 2),
        "LabTroisBGainMax": nombre(deux.gain_maximal_du_premier(), 0),
        "LabTroisBSecond": nombre(
            budget.bruit_du_second_amplificateur("AMP2") * _MICRO_CARRE * 1e5, 1
        ),
        "LabTroisBFiltre": nombre(budget.bruit_du_filtre_en_second() * _MICRO_CARRE, 2),
        "LabTroisBChaineAmpli": nombre(chaine["AMP1"] * 1e6, 2),
        "LabTroisBChaineFiltre": nombre(chaine["filtre"], 2),
        "LabTroisBChaineTotal": nombre(sum(chaine.values()), 2),
        "LabTroisBChaineEfficace": nombre(budget.bruit_de_la_chaine() * 1e6, 2),
        "LabTroisBCout": nombre(budget.cout(), 1),
        "LabTroisBGain": nombre(deux.gain(), 0),
        "LabTroisBGainSecond": nombre(deux.gain() / deux.gain_maximal_du_premier(), 0),
        "LabTroisBSource": nombre(UnDeux.SOURCE, 0),
        "LabTroisBPireCourant": nombre(deux.pire_courant_de_bruit() * 1e15, 0),
        "LabTroisBBande": nombre(UnDeux.BANDE, 0),
    }


def _troisieme() -> dict[str, str]:
    ampli, haut, bis, bas, _ = (p * _MICRO_CARRE for p in trois.puissances().values())
    budgets = [b * _MICRO_CARRE for b in trois.budgets()]
    return {
        "LabTroisCCourant": nombre(trois.pire_bruit_en_courant() * _MICRO_CARRE, 3),
        "LabTroisCTete": nombre(trois.bruit_efficace_en_tete(1e-6) * 1e6, 2),
        "LabTroisCAmpli": nombre(ampli, 4),
        "LabTroisCHaut": nombre(haut, 2),
        "LabTroisCBis": nombre(bis * 1e4, 2),
        "LabTroisCBas": nombre(bas, 2),
        "LabTroisCBasSecond": nombre(
            trois.bruit_du_passe_bas_en_second() * _MICRO_CARRE, 0
        ),
        "LabTroisCBudgetB": nombre(budgets[1], 2),
        "LabTroisCBudgetC": nombre(budgets[2], 2),
        "LabTroisCBudgetE": nombre(budgets[4], 2),
        "LabTroisCBudgetF": nombre(budgets[5], 2),
        "LabTroisCEfficace": nombre(trois.bruit_total() * 1e6, 2),
        "LabTroisCCadence": nombre(trois.cadence_totale() / 1e3, 0),
    }


def _quatrieme() -> dict[str, str]:
    large = quatre.bruits_a_l_entree(bande_apres_le_filtre=15e3)
    return {
        "LabTroisDFiltre": nombre(quatre.bruit_du_filtre_en_tete() * 1e3, 0),
        "LabTroisDTotal": nombre(quatre.bruit_total() * 1e6, 2),
        "LabTroisDMux": nombre(large[quatre.multiplexeur()] * 1e6, 2),
        "LabTroisDCan": nombre(large["CAN"] * 1e6, 2),
        "LabTroisDLarge": nombre(
            quatre.bruit_total(bande_apres_le_filtre=15e3) * 1e6, 1
        ),
        "LabTroisDComposants": str(sum(quatre.composants().values())),
        "LabTroisDCadenceMax": nombre(quatre.cadence_maximale(), 0),
        "LabTroisDVoies": str(quatre.voies_par_convertisseur()),
        "LabTroisDArret": nombre(quatre.bande_d_arret(), 0),
        "LabTroisDCadenceVoie": nombre(quatre.cadence_reelle_par_voie(), 0),
    }


def macros_du_labo_trois() -> dict[str, str]:
    """Toutes les valeurs que le corrige du laboratoire 3 compose."""
    return {**_premier(), **_deuxieme(), **_troisieme(), **_quatrieme()}
