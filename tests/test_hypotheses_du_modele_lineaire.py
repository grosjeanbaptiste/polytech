"""Chapitre I.2.2 et I.4 d'optimisation : hypotheses, remarques, cone des contraintes.

Sources : `Optimisation/SLIDES/Slides_01_Les_bases_de_la_programmation_lineaire.pdf`,
pages 31-32 (hypotheses d'additivite et de proportionnalite, couts non
proportionnels, cout fixe, gradient constant), page 26 (les ecarts sont
le temps non utilise) et page 37 (le cone des vecteurs `t_j`).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.combinaisons import est_combinaison_positive
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formes import forme_standard
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.hypotheses import (
    Hypothese,
    cout_a_charge_fixe,
    cout_unitaire_par_morceaux,
    gradient,
    hypotheses_violees,
    variable_de_charge,
)

X1, X2 = sp.symbols("x_1 x_2")
INF, SUP = Sens.INFERIEUR, Sens.SUPERIEUR


def test_une_fonction_lineaire_respecte_les_deux_hypotheses() -> None:
    assert hypotheses_violees(3 * X1 - 2 * X2, (X1, X2)) == frozenset()


def test_un_produit_de_deux_activites_viole_l_additivite() -> None:
    """Les activites interagissent : le gain n'est plus une somme sur `j`."""
    assert hypotheses_violees(X1 * X2, (X1, X2)) == {Hypothese.ADDITIVITE}


def test_une_puissance_viole_la_proportionnalite() -> None:
    assert hypotheses_violees(X1**2 + X2, (X1, X2)) == {Hypothese.PROPORTIONNALITE}


def test_un_terme_peut_violer_les_deux() -> None:
    violees = hypotheses_violees(X1**2 * X2, (X1, X2))
    assert violees == {Hypothese.ADDITIVITE, Hypothese.PROPORTIONNALITE}


def test_une_fonction_non_polynomiale_viole_la_proportionnalite() -> None:
    assert Hypothese.PROPORTIONNALITE in hypotheses_violees(sp.sqrt(X1), (X1, X2))


def test_le_gradient_de_la_fonction_economique_est_constant() -> None:
    """`dz/dx_j = c_j` pour tout `j` : il ne depend d'aucune variable."""
    programme = ProgrammeGeneral(Optimisation.MAX, (6, 5), ())
    assert gradient(programme) == (6, 5)


@pytest.mark.parametrize(
    ("niveau", "attendu"),
    [(0, 10), (5, 10), (6, 8), (20, 8), (21, 5), (100, 5)],
)
def test_le_cout_unitaire_change_aux_seuils(niveau: int, attendu: int) -> None:
    """`c = 10` jusqu'a 5, `8` jusqu'a 20, `5` au-dela (bornes superieures incluses)."""
    assert (
        cout_unitaire_par_morceaux(niveau, seuils=(5, 20), couts=(10, 8, 5)) == attendu
    )


def test_il_faut_un_cout_de_plus_que_de_seuils() -> None:
    with pytest.raises(ValueError, match="un cout de plus que de seuils"):
        cout_unitaire_par_morceaux(1, seuils=(5, 20), couts=(10, 8))


def test_le_cout_fixe_n_est_paye_que_si_l_activite_tourne() -> None:
    """`c = 0` si `x = 0`, `a + b x` sinon."""
    assert cout_a_charge_fixe(0, fixe=100, proportionnel=3) == 0
    assert cout_a_charge_fixe(4, fixe=100, proportionnel=3) == 112


def test_la_variable_binaire_vaut_un_des_que_l_activite_tourne() -> None:
    """`c = a y + b x`, avec `y = 0` si `x = 0` et `y = 1` si `x > 0`."""
    assert variable_de_charge(0) == 0
    assert variable_de_charge(sp.Rational(1, 100)) == 1
    for x in (0, 4):
        assert (
            cout_a_charge_fixe(x, fixe=100, proportionnel=3)
            == 100 * variable_de_charge(x) + 3 * x
        )


def test_un_niveau_negatif_est_refuse() -> None:
    with pytest.raises(ValueError, match="niveau d'activite negatif"):
        variable_de_charge(-1)


def test_l_ecart_d_une_capacite_est_le_temps_non_utilise() -> None:
    """Page 26 : machine de 500 unites, occupee `4 x1 + 2 x3 + 4 x5`."""
    machine = Contrainte((4, 0, 2, 0, 4), INF, 500)
    assert machine.ecart((50, 0, 100, 0, 0)) == 500 - (200 + 200)


def test_l_ecart_d_une_obligation_est_ce_qui_la_depasse() -> None:
    assert Contrainte((1, 1), SUP, 8).ecart((6, 5)) == 3


def _colonnes(sens: tuple[Sens, Sens, Sens]) -> tuple[list[sp.Matrix], sp.Matrix]:
    lignes = (((1, 1), 8), ((-2, 3), 6), ((1, -1), 2))
    contraintes = tuple(
        Contrainte(a, s, d) for (a, d), s in zip(lignes, sens, strict=True)
    )
    forme = forme_standard(ProgrammeGeneral(Optimisation.MAX, (6, 5), contraintes))
    colonnes = [forme.matrice[:, j] for j in range(forme.matrice.cols)]
    return colonnes, forme.seconds_membres


def test_d_est_dans_le_cone_des_colonnes_quand_le_domaine_n_est_pas_vide() -> None:
    """Page 37 : condition necessaire et suffisante d'existence d'une solution."""
    colonnes, d = _colonnes((INF, INF, INF))
    assert est_combinaison_positive(d, colonnes)


def test_d_est_hors_du_cone_quand_le_domaine_est_vide() -> None:
    """Figure I.8 : `D` est vide."""
    colonnes, d = _colonnes((INF, SUP, SUP))
    assert not est_combinaison_positive(d, colonnes)
