"""Syllabus d'electricite generale, chapitre 6 : le champ electromoteur.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 6-1 a 6-11, lues sur les pages rendues.

Le chapitre ne chiffre que ses sources (6.8 et 6.9). Les forces
electromotrices de contact y restent litterales : la loi des chaines est
donc eprouvee sur des valeurs arbitraires, ce qu'elle affirme etant
justement qu'elles n'importent pas.

Les piles relevent de la chimie : elles vivent dans `polytech.chemistry`,
et ce cours les importe.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.chemistry.domain.piles import (
    ACCUMULATEURS,
    PILES,
    fem_d_un_empilement,
)
from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.electrostatique.double_couche import (
    saut_de_potentiel,
)
from polytech.electromagnetisme.domain.force_electromotrice import (
    ChaineIsotherme,
    fem_le_long_d_un_trajet,
    moment_de_la_double_couche,
)
from polytech.electromagnetisme.domain.symboles import CARTESIENNES
from polytech.electromagnetisme.domain.thermoelectricite import (
    PLAGE_DU_COUPLE_CUIVRE_CONSTANTAN,
    fem_du_couple_cuivre_constantan,
    sensibilite_du_couple_cuivre_constantan,
    temperature_de_la_soudure_chaude,
    tension_de_seebeck,
)

X, Y, Z = CARTESIENNES
METAUX = ChaineIsotherme(
    potentiels_internes={"cuivre": 0.30, "zinc": -0.45, "fer": 0.05, "argent": 0.42},
    fems_de_surface={"cuivre": 1.10, "zinc": 0.70, "fer": 0.95, "argent": 1.25},
)
"""Des valeurs arbitraires : la loi des chaines affirme qu'elles n'importent pas."""


# --- 6.2 et 6.3 Force electromotrice, equilibre ---------------------------------


def test_la_fem_est_la_circulation_du_champ_electromoteur() -> None:
    """(6.2.1) : E_12 = integrale de 1 a 2 de E_e . dl."""
    parametre = sp.Symbol("t", real=True)
    champ = (sp.Integer(3), sp.Integer(0), sp.Integer(0))
    droite = (2 * parametre, parametre, sp.Integer(0))
    assert fem_le_long_d_un_trajet(champ, droite, parametre, 0, 1) == 6


def test_a_l_equilibre_la_fem_vaut_la_difference_de_potentiel_inverse() -> None:
    """(6.3.2) et (6.3.3) : E_e = grad V, donc E_12 = V_2 - V_1 = U_21."""
    parametre = sp.Symbol("t", real=True)
    potentiel = X**2 * Y + Z
    champ = tuple(sp.diff(potentiel, axe) for axe in CARTESIENNES)
    courbe = (parametre, parametre**2, 2 * parametre)
    fem = fem_le_long_d_un_trajet(champ, courbe, parametre, 0, 1)
    depart, arrivee = (
        potentiel.subs({X: 0, Y: 0, Z: 0}),
        potentiel.subs({X: 1, Y: 1, Z: 2}),
    )
    assert fem == arrivee - depart


def test_a_l_equilibre_la_fem_ne_depend_pas_du_chemin() -> None:
    """Page 6-3 : « elle ne depend pas du chemin d'integration »."""
    parametre = sp.Symbol("t", real=True)
    champ = tuple(sp.diff(X**2 * Y + Z, axe) for axe in CARTESIENNES)
    courbe = (parametre, parametre**2, 2 * parametre)
    corde = (parametre, parametre, 2 * parametre)
    assert fem_le_long_d_un_trajet(champ, courbe, parametre, 0, 1) == (
        fem_le_long_d_un_trajet(champ, corde, parametre, 0, 1)
    )


# --- 6.4 Effet Volta -----------------------------------------------------------


def test_la_fem_de_contact_est_la_difference_des_potentiels_interieurs() -> None:
    """(6.4.1) : E_12 = U_21 = V_2,int - V_1,int."""
    assert math.isclose(METAUX.fem_de_contact("zinc", "cuivre"), 0.75)
    assert METAUX.fem_de_contact("cuivre", "zinc") == -METAUX.fem_de_contact(
        "zinc", "cuivre"
    )
    assert METAUX.fem_de_contact("fer", "fer") == 0


def test_la_double_couche_de_contact_a_le_moment_qui_donne_la_fem() -> None:
    """(6.4.2) : E_12 = P_S / eps_0 -- la relation (2.16.5), lue dans l'autre sens."""
    fem = sp.Symbol("E_12", positive=True)
    moment = moment_de_la_double_couche(fem)
    assert moment == PERMITTIVITE * fem
    assert saut_de_potentiel(moment) == fem


def test_un_metal_inconnu_est_refuse() -> None:
    """La chaine ne connait que les metaux dont on lui a donne le potentiel."""
    with pytest.raises(ValueError, match="conducteur inconnu de la chaine"):
        METAUX.fem_de_contact("zinc", "plomb")
    with pytest.raises(ValueError, match="conducteur inconnu de la chaine"):
        METAUX.tension_voltaique(["plomb", "zinc"])


# --- 6.6 et 6.7 Chaines --------------------------------------------------------


def test_la_fem_d_une_chaine_est_la_somme_des_fem_de_contact() -> None:
    """(6.7.2) : E_1n = E_12 + E_23 + ... + E_(n-1)n."""
    chaine = ["zinc", "fer", "argent", "cuivre"]
    somme = (
        METAUX.fem_de_contact("zinc", "fer")
        + METAUX.fem_de_contact("fer", "argent")
        + METAUX.fem_de_contact("argent", "cuivre")
    )
    assert math.isclose(METAUX.fem(chaine), somme)


def test_la_loi_des_chaines_metalliques_isothermes() -> None:
    """Page 6-8 : comme « si les metaux extremes etaient en contact direct »."""
    directe = METAUX.fem(["zinc", "cuivre"])
    assert math.isclose(METAUX.fem(["zinc", "fer", "argent", "cuivre"]), directe)
    assert math.isclose(METAUX.fem(["zinc", "argent", "cuivre"]), directe)


def test_une_chaine_isotherme_fermee_n_a_pas_de_fem() -> None:
    """Page 6-8 : sinon un courant circulerait sans aucune source d'energie."""
    fermee = ["zinc", "fer", "argent", "cuivre", "zinc"]
    assert math.isclose(METAUX.fem(fermee), 0.0, abs_tol=1e-12)


def test_une_chaine_d_un_seul_conducteur_n_a_pas_de_contact() -> None:
    """Une chaine relie au moins deux conducteurs."""
    with pytest.raises(ValueError, match="au moins deux conducteurs"):
        METAUX.fem(["zinc"])


def test_la_tension_voltaique_de_deux_metaux_dans_un_isolant() -> None:
    """(6.6.2) : V_2,ext - V_1,ext = E_01 + E_12 + E_20."""
    attendu = 0.70 + 0.75 - 1.10
    assert math.isclose(METAUX.tension_voltaique(["zinc", "cuivre"]), attendu)


def test_la_tension_voltaique_ne_depend_que_des_metaux_extremes() -> None:
    """(6.7.1) et le theoreme de la page 6-8."""
    directe = METAUX.tension_voltaique(["zinc", "cuivre"])
    assert math.isclose(
        METAUX.tension_voltaique(["zinc", "fer", "argent", "cuivre"]), directe
    )
    assert not math.isclose(METAUX.tension_voltaique(["zinc", "argent"]), directe)


# --- 6.8 et 6.9 Effet Seebeck, sources ------------------------------------------


def test_la_fem_du_thermocouple_cuivre_constantan() -> None:
    """(6.9.1) : E = 41 Theta_c + 0,05 Theta_c^2 microvolts, soudure froide a 0 degC."""
    assert fem_du_couple_cuivre_constantan(0) == 0
    assert math.isclose(fem_du_couple_cuivre_constantan(100), 4.6e-3)
    assert math.isclose(fem_du_couple_cuivre_constantan(300), 16.8e-3)


@pytest.mark.parametrize("temperature", [-1, 301])
def test_la_loi_du_thermocouple_ne_vaut_que_de_0_a_300_degres(
    temperature: float,
) -> None:
    """Page 6-11 : « on a (de 0 a 300 degC) »."""
    with pytest.raises(ValueError, match="hors de l'intervalle de 0 a 300"):
        fem_du_couple_cuivre_constantan(temperature)


def test_la_sensibilite_du_thermocouple_recoupe_le_paragraphe_6_8() -> None:
    """6.8 dit « 40 a 50 microvolts par degre » : c'est dE/dTheta entre 0 et 90 degC."""
    bas, haut = PLAGE_DU_COUPLE_CUIVRE_CONSTANTAN
    assert math.isclose(sensibilite_du_couple_cuivre_constantan(0), 41e-6)
    assert bas <= sensibilite_du_couple_cuivre_constantan(0) <= haut
    assert math.isclose(sensibilite_du_couple_cuivre_constantan(90), haut)
    assert sensibilite_du_couple_cuivre_constantan(300) > haut


def test_la_sensibilite_est_la_derivee_de_la_fem() -> None:
    """La pente de (6.9.1), par difference finie centree."""
    pas = 1e-3
    pente = (
        fem_du_couple_cuivre_constantan(150 + pas)
        - fem_du_couple_cuivre_constantan(150 - pas)
    ) / (2 * pas)
    assert math.isclose(
        pente, sensibilite_du_couple_cuivre_constantan(150), rel_tol=1e-9
    )


def test_le_modele_lineaire_sous_estime_la_fem_a_chaud() -> None:
    """Le terme quadratique : 0,5 mV de plus que 41 microvolts par degre, a 100 degC."""
    lineaire = tension_de_seebeck(41e-6, 100)
    assert math.isclose(fem_du_couple_cuivre_constantan(100) - lineaire, 0.5e-3)


@pytest.mark.parametrize("temperature", [0.0, 37.0, 100.0, 300.0])
def test_la_temperature_se_retrouve_depuis_la_fem(temperature: float) -> None:
    """Page 6-10 : un convertisseur temperature - f.e.m., lu a l'envers."""
    fem = fem_du_couple_cuivre_constantan(temperature)
    assert math.isclose(
        temperature_de_la_soudure_chaude(fem), temperature, abs_tol=1e-9
    )


def test_une_fem_hors_de_la_plage_ne_donne_pas_de_temperature() -> None:
    """Au-dela de 16,8 mV, ou en dessous de zero, la loi (6.9.1) ne dit rien."""
    with pytest.raises(ValueError, match="hors de la plage du couple"):
        temperature_de_la_soudure_chaude(17e-3)
    with pytest.raises(ValueError, match="hors de la plage du couple"):
        temperature_de_la_soudure_chaude(-1e-6)


def test_les_piles_du_paragraphe_6_9() -> None:
    """Page 6-9 : Volta 0,85 V, Leclanche 1,5 V, mercure 1,35 V, Weston 1,018 6 V."""
    fems = {nom: pile.fem for nom, pile in PILES.items()}
    assert fems == {"Volta": 0.85, "Leclanche": 1.5, "mercure": 1.35, "Weston": 1.0186}
    assert PILES["Volta"].chaine == "Zn/H2SO4 aq/Cu"
    assert not any(pile.reversible for pile in PILES.values())


def test_la_pile_weston_sert_d_etalon() -> None:
    """Remarque de la page 6-10 : sa fem est « suffisamment stable »."""
    etalons = [nom for nom, pile in PILES.items() if pile.etalon]
    assert etalons == ["Weston"]


def test_les_accumulateurs_sont_des_piles_reversibles() -> None:
    """Page 6-10 : cadmium/nickel 1,3 V, plomb 2 V."""
    fems = {nom: pile.fem for nom, pile in ACCUMULATEURS.items()}
    assert fems == {"cadmium-nickel": 1.3, "plomb": 2.0}
    assert all(pile.reversible for pile in ACCUMULATEURS.values())


def test_les_elements_en_serie_additionnent_leurs_fem() -> None:
    """Page 6-9 : la pile de Volta de 1800, un empilement d'elements identiques."""
    assert math.isclose(fem_d_un_empilement(PILES["Volta"], 10), 8.5)
    assert fem_d_un_empilement(ACCUMULATEURS["plomb"], 6) == 12.0


def test_un_empilement_compte_au_moins_un_element() -> None:
    """Zero rondelle ne fait pas une pile."""
    with pytest.raises(ValueError, match="au moins un element"):
        fem_d_un_empilement(PILES["Volta"], 0)
