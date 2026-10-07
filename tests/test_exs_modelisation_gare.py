"""Optimisation : la modelisation, problemes IIIa, IIIb et IIIc (la gare).

Source : `Optimisation/EXS/Exs_Modelisation.pdf`, planches 12 a 18. Six
periodes de quatre heures, des equipes de huit heures. Les modeles des
planches sont resolus ; celui de la planche 18 porte deux coquilles, que
la relecture de l'enonce met en evidence.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.application import modele_gare as ga
from polytech.optimisation.application import modele_gare_categories as ca

Q = sp.Rational

# --- Probleme IIIa (planches 12 a 14) ----------------------------------------


def test_iiia_les_heures_payees_de_chaque_equipe() -> None:
    """Planche 14 : la nuit (20 h - 8 h) compte double. 16, 12, 8, 8, 12, 16."""
    assert tuple(ga.heures_payees(debut) for debut in range(1, 7)) == ga.SALAIRES
    assert ga.SALAIRES == (16, 12, 8, 8, 12, 16)


def test_iiia_le_plus_petit_effectif() -> None:
    assert ga.optimum(ga.BESOINS, "effectif")[0] == 27


def test_iiia_la_borne_de_l_effectif_sans_simplexe() -> None:
    """Second chemin : les periodes impaires, a elles seules, demandent 27 personnes."""
    assert ga.besoin_des_periodes(ga.BESOINS, (1, 3, 5)) == 27
    assert ga.besoin_des_periodes(ga.BESOINS, (2, 4, 6)) == 21


def test_iiia_le_plus_petit_cout_salarial() -> None:
    cout, equipes = ga.optimum(ga.BESOINS, "salaire")
    assert cout == 280
    assert ga.est_couvert(ga.BESOINS, equipes)
    assert sum(equipes) == 27


def test_iiia_une_equipe_couvre_sa_periode_et_la_suivante() -> None:
    """Celle de 20 h couvre la derniere periode et la premiere du lendemain."""
    assert ga.presents((0, 0, 0, 0, 0, 4)) == (4, 0, 0, 0, 0, 4)
    assert ga.presents((1, 2, 0, 0, 0, 0)) == (1, 3, 2, 0, 0, 0)
    assert not ga.est_couvert(ga.BESOINS, (3, 5, 5, 3, 10, 0))


# --- Probleme IIIb (planches 15 et 16) ---------------------------------------


def test_iiib_les_coefficients_de_la_planche_16() -> None:
    """Dans l'ordre des lignes de la planche : periodes 2, 3, 4, 5, 6 puis 1."""
    assert ga.ecarts(ga.VARIANTES[0]) == (1, -1, 1, -2, 2, 1)
    assert ga.ecarts(ga.VARIANTES[1]) == (2, 2, 4, -2, -1, -1)


def test_iiib_le_cout_de_chaque_definition_des_besoins() -> None:
    assert ga.cout_par_definition() == (280, 280, 272)


def test_iiib_le_modele_a_deux_binaires_choisit_la_troisieme() -> None:
    """Second chemin : le modele de la planche 16, resolu d'un seul tenant."""
    cout, delta, gamma = ga.choix_du_modele()
    assert (cout, delta, gamma) == (272, 0, 1)


# --- Probleme IIIc (planche 18) ----------------------------------------------


def test_iiic_les_couts_des_horaires_coupes() -> None:
    """Deux fois quatre heures, majorees de 10 % : 13,2 partout sauf deux."""
    assert ca.cout_d_un_horaire_coupe(2, 6) == Q(88, 5)
    assert ca.cout_d_un_horaire_coupe(3, 5) == Q(44, 5)
    autres = [p for p in ca.PAIRES if p not in {(2, 6), (3, 5)}]
    assert {ca.cout_d_un_horaire_coupe(*p) for p in autres} == {Q(66, 5)}


def test_iiic_la_planche_ecrit_13_3_pour_un_horaire_qui_vaut_13_2() -> None:
    assert ca.COUT_IMPRIME[(2, 4)] == Q(133, 10)
    assert ca.cout_d_un_horaire_coupe(2, 4) == Q(132, 10)


def test_iiic_la_derniere_ligne_de_la_planche_oublie_la_premiere_periode() -> None:
    """Elle repete `s24 + s25 + s26` ; ce sont `s13`, `s14`, `s15` qui y travaillent."""
    assert ca.horaires_de_la_periode(1) == ((1, 3), (1, 4), (1, 5))
    assert ca.LIGNE_IMPRIMEE_DE_LA_PERIODE_1 == ((2, 4), (2, 5), (2, 6))


def test_iiic_le_modele_imprime_laisse_la_gare_vide_la_nuit() -> None:
    """Un plan admissible pour la planche, moins cher, sans personne de 0 h a 4 h."""
    plan = ca.CONTRE_EXEMPLE
    assert ca.respecte(ca.programme(corrige=False), plan)
    assert not ca.respecte(ca.programme(corrige=True), plan)
    assert ca.presents(plan)[0] == 0
    assert ca.cout(plan) == Q(1248, 5)


def test_iiic_l_optimum_du_modele_corrige() -> None:
    cout, plan = ca.optimum()
    assert cout == Q(1334, 5)
    assert all(p >= b for p, b in zip(ca.presents(plan), ga.BESOINS, strict=True))
    assert sum(plan) <= 30


def test_iiic_les_horaires_coupes_font_gagner_malgre_les_categories() -> None:
    """Huit heures d'affilee coutaient 280 : deux fois quatre collent aux besoins."""
    assert ca.optimum()[0] < ga.optimum(ga.BESOINS, "salaire")[0]


def test_un_objectif_inconnu_est_refuse() -> None:
    with pytest.raises(KeyError, match="objectif inconnu : "):
        ga.programme(ga.BESOINS, "confort")


def test_un_modele_sans_solution_entiere_est_refuse() -> None:
    """Demander plus de monde que les trente personnes : aucune solution."""
    with pytest.raises(ValueError, match="aucune solution entiere"):
        ga.en_entiers(ca.programme_des_besoins((40, 40, 40, 40, 40, 40)))
