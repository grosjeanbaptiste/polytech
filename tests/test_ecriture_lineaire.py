"""Ecrire une expression lineaire comme le cours l'imprime.

`7 u_1 - 4 u_2 + 5 u_3` : c'est la forme des pages 92-93 du chapitre IV
d'optimisation, dont les explications de la dualite ont besoin.
"""

from __future__ import annotations

from polytech.optimisation.domain.contraintes import Contrainte, Sens
from polytech.optimisation.domain.ecriture import ecrire_contrainte, ecrire_lineaire


def test_les_signes_se_lient_et_le_coefficient_un_se_tait() -> None:
    assert ecrire_lineaire((7, -4, 5), "u") == "7 u_1 - 4 u_2 + 5 u_3"
    assert ecrire_lineaire((1, -3, 1), "u") == "u_1 - 3 u_2 + u_3"


def test_un_terme_nul_est_omis_sans_decaler_les_indices() -> None:
    assert ecrire_lineaire((0, 2, 1), "u") == "2 u_2 + u_3"


def test_un_premier_terme_negatif_porte_son_signe_colle() -> None:
    assert ecrire_lineaire((-3, 2), "x") == "-3 x_1 + 2 x_2"
    assert ecrire_lineaire((-1, -1), "x") == "-x_1 - x_2"


def test_une_expression_nulle_s_ecrit_zero() -> None:
    assert ecrire_lineaire((0, 0), "x") == "0"


def test_une_contrainte_s_ecrit_avec_son_sens_et_son_second_membre() -> None:
    contrainte = Contrainte((0, 2, 1), Sens.EGAL, -4)
    assert ecrire_contrainte(contrainte, "u") == "2 u_2 + u_3 = -4"
