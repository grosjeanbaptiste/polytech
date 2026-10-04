"""Chapitre I.2.1 d'optimisation : les trois modeles classiques.

Oracles : les illustrations du cours
(`Optimisation/SLIDES/Slides_01_Les_bases_de_la_programmation_lineaire.pdf`,
pages 26 a 30) -- la planification a quatre machines et cinq produits,
le melange de charbons, le transport de cinq depots vers six centres.
Chaque programme construit est compare a celui que le cours ecrit.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.modeles import (
    contrainte_de_teneur,
    planification_de_production,
    probleme_de_melange,
)
from polytech.optimisation.domain.transport import ProblemeDeTransport

INF, SUP, EGA = Sens.INFERIEUR, Sens.SUPERIEUR, Sens.EGAL

TEMPS = ((4, 0, 2, 0, 4), (2, 3, 0, 0, 3), (3, 0, 1, 4, 0), (0, 5, 4, 2, 0))
COUTS = (
    (9, 12, 9, 6, 9, 10),
    (6, 8, 9, 11, 3, 11),
    (7, 3, 11, 2, 3, 10),
    (6, 8, 7, 10, 3, 5),
    (5, 6, 9, 2, 7, 3),
)
STOCKS, DEMANDES = (7, 6, 8, 10, 8), (9, 6, 7, 8, 4, 5)


def test_la_planification_de_la_page_26() -> None:
    programme = planification_de_production(
        gains=(100, 75, 50, 25, 50),
        capacites=(500, 800, 700, 900),
        temps=TEMPS,
    )
    assert programme.optimisation is Optimisation.MAX
    assert programme.objectif == (100, 75, 50, 25, 50)
    assert programme.contraintes[0] == Contrainte((4, 0, 2, 0, 4), INF, 500)
    assert programme.contraintes[3] == Contrainte((0, 5, 4, 2, 0), INF, 900)


def test_le_probleme_dietetique_minimise_un_cout_sous_des_besoins() -> None:
    """`min c x`, `T x >= d` : deux aliments, deux elements nutritifs."""
    programme = probleme_de_melange(
        prix=(3, 2), apports=((1, 2), (4, 1)), besoins=(8, 6)
    )
    assert programme.optimisation is Optimisation.MIN
    assert programme.contraintes == (
        Contrainte((1, 2), SUP, 8),
        Contrainte((4, 1), SUP, 6),
    )


def test_une_teneur_maximale_se_linearise() -> None:
    """`somme(c_i x_i) <= C somme(x_i)` devient `somme((c_i - C) x_i) <= 0`."""
    assert contrainte_de_teneur((12, 8, 15), 10, INF) == Contrainte((2, -2, 5), INF, 0)


def test_la_teneur_linearisee_dit_la_meme_chose_que_le_rapport() -> None:
    """Melange (1, 3, 0) : teneur `(12 + 24) / 4 = 9 <= 10`."""
    melange = (1, 3, 0)
    teneur = sp.Rational(12 * 1 + 8 * 3 + 15 * 0, sum(melange))
    contrainte = contrainte_de_teneur((12, 8, 15), 10, INF)
    assert contrainte.est_satisfaite(melange) is bool(teneur <= 10)


def test_un_temps_de_mauvaise_dimension_est_refuse() -> None:
    with pytest.raises(ValueError, match="une ligne de temps par machine"):
        planification_de_production(
            gains=(1, 2), capacites=(5,), temps=((1, 1), (2, 2))
        )


def _illustration() -> ProblemeDeTransport:
    return ProblemeDeTransport(stocks=STOCKS, demandes=DEMANDES, couts=COUTS)


def test_le_transport_a_rs_variables_et_r_plus_s_contraintes() -> None:
    programme = _illustration().programme()
    assert len(programme.objectif) == 5 * 6
    assert len(programme.contraintes) == 5 + 6


def test_la_fonction_economique_du_transport_est_celle_de_la_page_29() -> None:
    objectif = _illustration().programme().objectif
    assert objectif[:6] == (9, 12, 9, 6, 9, 10)
    assert objectif[-6:] == (5, 6, 9, 2, 7, 3)


def test_le_modele_general_du_transport_est_celui_de_la_page_28() -> None:
    """`somme_j x_ij <= q_i` pour les depots, `somme_i x_ij >= d_j` pour les centres."""
    contraintes = _illustration().programme().contraintes
    assert contraintes[0] == Contrainte((1,) * 6 + (0,) * 24, INF, 7)
    dernier_centre = tuple(1 if k % 6 == 5 else 0 for k in range(30))
    assert contraintes[-1] == Contrainte(dernier_centre, SUP, 5)


def test_les_contraintes_equilibrees_sont_celles_de_la_page_30() -> None:
    """`x11 + ... + x16 = 7` et `x16 + x26 + ... + x56 = 5`."""
    contraintes = _illustration().programme_equilibre().contraintes
    assert contraintes[0] == Contrainte((1,) * 6 + (0,) * 24, EGA, 7)
    assert contraintes[4] == Contrainte((0,) * 24 + (1,) * 6, EGA, 8)
    premier_centre = tuple(1 if k % 6 == 0 else 0 for k in range(30))
    assert contraintes[5] == Contrainte(premier_centre, EGA, 9)
    dernier_centre = tuple(1 if k % 6 == 5 else 0 for k in range(30))
    assert contraintes[-1] == Contrainte(dernier_centre, EGA, 5)


def test_chaque_variable_du_transport_intervient_deux_fois() -> None:
    """La structure de `T` : un `1` pour le depot, un `1` pour le centre."""
    matrice = _illustration().matrice_technologique()
    assert matrice.shape == (11, 30)
    assert all(sum(matrice[:, j]) == 2 for j in range(30))
    assert set(matrice) == {0, 1}


def test_l_illustration_est_equilibree() -> None:
    assert _illustration().surplus == 0
    assert _illustration().equilibre() == _illustration()


def test_un_centre_fictif_absorbe_le_surplus_a_cout_nul() -> None:
    excedentaire = ProblemeDeTransport(
        stocks=(7, 6), demandes=(5, 4), couts=((1, 2), (3, 4))
    )
    equilibre = excedentaire.equilibre()
    assert equilibre.demandes == (5, 4, 4)
    assert equilibre.couts == ((1, 2, 0), (3, 4, 0))
    assert equilibre.surplus == 0


def test_le_probleme_equilibre_s_ecrit_avec_des_egalites() -> None:
    contraintes = _illustration().programme_equilibre().contraintes
    assert {c.sens for c in contraintes} == {EGA}


def test_une_demande_superieure_aux_stocks_est_refusee() -> None:
    """Page 29 : la quantite disponible doit permettre de satisfaire la demande."""
    with pytest.raises(ValueError, match="demande totale"):
        ProblemeDeTransport(stocks=(1,), demandes=(5,), couts=((1,),)).equilibre()


def test_une_matrice_de_couts_mal_dimensionnee_est_refusee() -> None:
    with pytest.raises(ValueError, match="matrice des couts"):
        ProblemeDeTransport(stocks=(1, 2), demandes=(3,), couts=((1,),))
