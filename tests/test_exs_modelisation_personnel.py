"""Optimisation : la modelisation, problemes I, IV et V (du personnel a gerer).

Source : `Optimisation/EXS/Exs_Modelisation.pdf`, planches 4-5 et 19 a 22.
Les planches posent des modeles et ne les resolvent pas. Chaque modele
est ici recopie tel quel puis resolu, et son optimum recoupe par un
second chemin qui ne passe pas par le programme lineaire.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.application import modele_personnel as pe
from polytech.optimisation.application import modele_pilotes as pi
from polytech.optimisation.application import modele_radios as ra
from polytech.optimisation.application.modele_variables import Variables

Q = sp.Rational

# --- Les variables nommees ---------------------------------------------------


def test_une_ligne_place_chaque_coefficient_sous_son_nom() -> None:
    variables = Variables(("T1", "T2", "E1"))
    assert variables.ligne({"E1": -1, "T1": 3}) == (3, 0, -1)
    assert variables.lire((7, 8, 9), "T") == (7, 8)
    with pytest.raises(KeyError, match="variable inconnue : "):
        variables.ligne({"X1": 1})


# --- Probleme I : le personnel de bord (planches 4 et 5) ---------------------


def test_i_le_modele_de_la_planche_5() -> None:
    """Treize bilans d'effectif, douze besoins en heures, douze plafonds."""
    programme = pe.programme()
    assert len(programme.objectif) == 12 + 13 + 13 + 12
    assert len(programme.contraintes) == 13 + 12 + 12
    coefficients = programme.contraintes[12].coefficients
    retour = dict(zip(pe.VARIABLES.noms, coefficients, strict=True))
    assert {nom: c for nom, c in retour.items() if c} == {"T12": 1, "E13": 1, "L13": -1}
    assert programme.contraintes[12].second_membre == 65


def test_i_le_plan_optimal() -> None:
    """Le cout minimal, et un plan qui l'atteint : il est entier d'emblee."""
    plan = pe.plan()
    assert plan.cout == 2_600_000
    assert plan.effectifs == (60, 60, 70, 60, 53, 48, 80, 100, 85, 80, 75, 70)
    assert sum(plan.heures_supplementaires) == 0
    assert sum(plan.licenciements) == 22


def test_i_le_plan_respecte_l_enonce() -> None:
    """Relu sur l'enonce, sans le programme : besoins couverts, retour a 60."""
    plan = pe.plan()
    for effectif, besoin in zip(plan.effectifs, pe.BESOINS, strict=True):
        assert 150 * effectif >= besoin
    assert plan.effectifs[-1] - 5 + plan.embauches[-1] - plan.licenciements[-1] == 60
    assert 3000 * sum(plan.effectifs) + 3500 * sum(plan.licenciements) == plan.cout


def test_i_le_meme_cout_par_recurrence() -> None:
    """Second chemin : mois par mois, sur tous les effectifs entiers possibles."""
    assert pe.cout_par_recurrence() == 2_600_000


# --- Probleme IV : les pilotes (planches 19 et 20) ---------------------------


@pytest.mark.parametrize(
    ("semaines", "relaxation"),
    [
        (2, 61305),
        (3, Q(180590, 3)),
        (4, Q(119285, 2)),
        (5, 60500),
        (6, 62100),
        (7, 63700),
    ],
)
def test_iv_la_relaxation_selon_le_nombre_de_semaines(
    semaines: int, relaxation: sp.Expr
) -> None:
    assert pi.relaxation(semaines) == relaxation


def test_iv_le_cout_ne_depend_que_des_instructeurs() -> None:
    """`56000 + 5 (somme des effectifs - 700) + 195 (instructeurs)`."""
    assert pi.cout((Q(3, 2), 0, 0, 0)) == Q(119285, 2)
    assert pi.cout((2, 0, 0, 0)) == 59890
    assert pi.cout((8, 0, 0)) == 60460
    assert pi.cout((0, 0, 0, 0, 0)) == 60500


def test_iv_un_plan_entier_a_quatre_semaines() -> None:
    """Deux instructeurs la premiere semaine : 700 semaines de vol, 59890."""
    plan = pi.PLAN_A_QUATRE
    assert pi.est_admissible(plan)
    assert sum(plan.vols) == 700
    assert pi.cout_du_plan(plan) == 59890
    assert not pi.est_admissible(pi.plan_decale(plan, 1, 200))
    assert not pi.est_admissible(pi.plan_decale(plan, 4, 181))
    assert not pi.est_admissible(pi.plan_decale(plan, 1, 161))


def test_iv_aucun_choix_d_instructeurs_moins_cher_ne_suffit() -> None:
    """Second chemin : tout choix entier moins cher vole moins de 700 semaines."""
    moins_chers = pi.choix_moins_chers(4, 59890)
    assert (1, 1, 0, 0) in moins_chers
    assert (0, 0, 0, 5) in moins_chers
    assert (2, 0, 0, 0) not in moins_chers
    assert all(pi.vols_possibles(4, choix) < 700 for choix in moins_chers)


def test_iv_sans_instructeur_il_faut_cinq_semaines() -> None:
    """Un equipage vole une semaine sur deux : 160 par semaine au plus."""
    assert pi.vols_possibles(4, (0, 0, 0, 0)) == 640
    assert pi.vols_possibles(5, (0, 0, 0, 0, 0)) == 800


# --- Probleme V : les radios (planches 21 et 22) -----------------------------


def test_v_le_plan_optimal_perd_de_l_argent() -> None:
    """Le « profit » maximal est negatif : le contrat coute 63000."""
    plan = ra.plan()
    assert plan.profit == -63000
    assert plan.travailleurs == (10, 130, 130, 130)
    assert plan.instructeurs == (30, 0, 0)
    assert plan.eleves == (90, 0, 0)


def test_v_le_profit_relu_sur_l_enonce() -> None:
    """Second chemin : recettes, salaires et formation, sans l'objectif du modele."""
    plan = ra.plan()
    assert ra.radios(plan) == 20000
    assert ra.recettes_nettes(plan) == 222000
    assert ra.salaires(plan) == 258000 + 27000
    assert ra.recettes_nettes(plan) - ra.salaires(plan) == plan.profit


def test_v_la_solution_entiere_est_celle_de_la_relaxation() -> None:
    assert ra.profit_en_entiers() == ra.plan().profit
