"""Optimisation : la modelisation, les onze « modelisations particulieres ».

Source : `Optimisation/EXS/Exs_Modelisation.pdf`, planches 47 a 69. Un
plan de production sur cinq mois, et onze facons de le contraindre.
Deux chemins independants : tous les plans entiers, juges sur l'enonce ;
puis le modele de chaque planche, resolu par « Branch and Bound ».
"""

from __future__ import annotations

import pytest

from polytech.optimisation.application import modele_production as pr
from polytech.optimisation.application import modele_production_binaires as bi
from polytech.optimisation.application import modele_production_croissante as cr
from polytech.optimisation.application import modele_production_planches as pp
from polytech.optimisation.domain.algorithme import Issue

LIBRE = (61, (11, 0, 14, 0, 5))
OPTIMUMS = {
    "I": (30, (30, 0, 0, 0, 0)),
    "III": (85, (7, 6, 6, 6, 5)),
    "IV": (82, (7, 5, 7, 6, 5)),
    "V": (71, (6, 5, 14, 0, 5)),
    "VI": LIBRE,
    "VII": LIBRE,
    "VIII": LIBRE,
    "IX": LIBRE,
    "X": LIBRE,
}

# --- Les plans, juges sur l'enonce -------------------------------------------


def test_le_stock_est_la_somme_des_avances_sur_la_demande() -> None:
    """Planche 50 : `4 x1 + 3 x2 + 2 x3 + x4 - 60`."""
    plan = (11, 0, 14, 0, 5)
    assert pr.stock(plan) == 6 + 0 + 6 + 0
    assert pr.stock(plan) == 4 * 11 + 3 * 0 + 2 * 14 + 0 - 60


def test_le_cout_d_un_plan_variante_par_variante() -> None:
    """Un plan qui suit la demande : 79 de production, aucun stock."""
    suivi = (5, 6, 8, 6, 5)
    assert pr.cout("stock", suivi) == 79
    assert pr.cout("III", suivi) == 79 + 4 * 1 + 4 * 2 + 2 * 2 + 2 * 1
    assert pr.cout("II", (5, 7, 7, 7, 7)) == 89 + 2 + 2 * 2
    assert pr.cout("IV", suivi) == 79 + 2 * (8 - 5)


def test_tous_les_plans_servent_la_demande_a_temps() -> None:
    plans = pr.plans()
    assert len(plans) == 10339
    assert (4, 7, 8, 6, 5) not in plans
    assert all(sum(plan) == 30 for plan in plans)


@pytest.mark.parametrize("variante", sorted(OPTIMUMS))
def test_l_optimum_de_chaque_variante(variante: str) -> None:
    assert pr.meilleur(variante) == OPTIMUMS[variante]
    assert pr.ex_aequo(variante) == 1


def test_sans_condition_le_plan_libre_suffit_aux_variantes_vi_a_x() -> None:
    """Il fabrique trois mois exactement, et ni en fevrier ni en avril."""
    assert pr.meilleur("stock") == LIBRE


def test_ii_aucun_plan_ne_croit_sans_cesse() -> None:
    """19 unites en trois mois croissants : au moins 7 en mars, donc 33 en tout."""
    with pytest.raises(ValueError, match="aucun plan ne respecte la variante "):
        pr.meilleur("II")


def test_xi_la_demande_d_origine_coute_moins() -> None:
    """61 contre 62 : la demande alternative ne vaut pas le changement."""
    assert pr.meilleur_pour(pr.DEMANDES) == LIBRE
    assert pr.meilleur_pour(pr.ALTERNATIVE) == (62, (12, 0, 12, 0, 6))


def test_une_variante_inconnue_est_refusee() -> None:
    with pytest.raises(KeyError, match="variante inconnue : "):
        pr.meilleur("XII")


# --- Les modeles des planches ------------------------------------------------


@pytest.mark.parametrize("variante", ["I", "III", "IV", "V"])
def test_le_modele_de_la_planche_rend_le_meme_optimum(variante: str) -> None:
    assert pp.optimum(variante) == OPTIMUMS[variante]


@pytest.mark.parametrize("variante", ["VI", "VII", "VIII", "IX", "X"])
def test_le_modele_a_binaires_rend_le_meme_optimum(variante: str) -> None:
    assert bi.optimum(variante) == OPTIMUMS[variante]


def test_ii_le_modele_de_la_planche_50_est_impossible() -> None:
    assert pp.issue("II") is Issue.IMPOSSIBLE


def test_ii_il_devient_possible_si_l_on_peut_produire_plus_que_la_demande() -> None:
    """`(5, 7, 7, 7, 7)` : 33 unites au moins, trois de plus que la demande."""
    assert cr.plus_petit_total_croissant() == 33


def test_xi_la_planche_69_oublie_que_le_stock_depend_de_la_demande() -> None:
    """Sa constante reste 60 ; pour la demande alternative elle vaut 58."""
    assert pr.avances_cumulees(pr.DEMANDES) == 60
    assert pr.avances_cumulees(pr.ALTERNATIVE) == 58


def test_xi_l_oubli_renverse_la_decision() -> None:
    """Imprime : 60, demande alternative. Corrige : 61, demande d'origine."""
    assert bi.choix_de_la_demande(corrige=False) == (60, 1, (12, 0, 12, 0, 6))
    assert bi.choix_de_la_demande(corrige=True) == (61, 0, (11, 0, 14, 0, 5))


def test_xi_le_modele_corrige_chiffre_bien_la_demande_alternative() -> None:
    """`delta` impose a 1 : 62, comme la revue de tous les plans ; la planche dit 60."""
    assert bi.choix_de_la_demande(corrige=True, impose=1)[0] == 62
    assert bi.choix_de_la_demande(corrige=False, impose=1)[0] == 60
    assert bi.choix_de_la_demande(corrige=False, impose=0)[0] == 61


def test_une_variante_sans_modele_est_refusee() -> None:
    """Chaque module ne porte que les planches qu'il annonce."""
    with pytest.raises(KeyError, match="variante sans modele ici : "):
        pp.programme("VI")
    with pytest.raises(KeyError, match="variante sans modele ici : "):
        bi.programme("III")
