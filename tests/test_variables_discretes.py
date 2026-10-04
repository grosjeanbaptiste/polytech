"""Chapitre XIV.1 d'optimisation : ce que les variables discretes changent.

Oracles (`Optimisation/SLIDES/Slides_14_Introduction_a_la_programmation_`
`lineaire_en_variables_entieres.pdf`, pages 290 a 296) : l'illustration
`min -3 x1 - 2 x2` -- (LP) optimal en `(25/6, 4/3)` pour `-91/6`, (ILP) a
onze points, optimal en `(3, 2)` pour `-13`, (MILP) optimal en
`(4, 3/2)` pour `-15` ; le polyedre `D2`, optimal en `(7/2, 2)` pour
`-29/2` ; la fermeture convexe de la figure XIV.5.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.bases import sommets
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.entiers import (
    Nature,
    ProgrammeEntier,
    points_entiers,
)
from polytech.optimisation.domain.formes import forme_standard
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.resolution import resoudre
from polytech.optimisation.domain.unimodularite import est_totalement_unimodulaire

R = sp.Rational
INF = Sens.INFERIEUR
BOITE = ((0, 6), (0, 6))


def _lineaire(*supplementaires: Contrainte) -> ProgrammeGeneral:
    """Page 290 : `min -3 x1 - 2 x2` sous (a), (b), (c)."""
    contraintes = (
        Contrainte((-3, 4), INF, 6),
        Contrainte((2, 2), INF, 11),
        Contrainte((4, -5), INF, 10),
    )
    return ProgrammeGeneral(
        Optimisation.MIN, (-3, -2), (*contraintes, *supplementaires)
    )


def _fermeture_convexe() -> ProgrammeGeneral:
    """Page 294, figure XIV.5 : le polyedre dont les sommets sont des points de S."""
    contraintes = (
        Contrainte((-1, 1), INF, 1),
        Contrainte((1, 1), INF, 5),
        Contrainte((1, 0), INF, 3),
        Contrainte((1, -1), INF, 2),
    )
    return ProgrammeGeneral(Optimisation.MIN, (-3, -2), contraintes)


def test_la_nature_du_probleme_depend_des_variables_astreintes() -> None:
    assert ProgrammeEntier(_lineaire(), frozenset()).nature is Nature.LP
    assert ProgrammeEntier(_lineaire(), frozenset({0, 1})).nature is Nature.ILP
    assert ProgrammeEntier(_lineaire(), frozenset({0})).nature is Nature.MILP


def test_une_variable_entiere_hors_du_programme_est_refusee() -> None:
    with pytest.raises(ValueError, match="variable entiere hors du programme"):
        ProgrammeEntier(_lineaire(), frozenset({2}))


def test_la_relaxation_lineaire_est_optimale_en_un_sommet_non_entier() -> None:
    """Page 290 : `(25/6, 4/3)`, intersection de (b) et (c), `z = -91/6`."""
    relaxation = ProgrammeEntier(_lineaire(), frozenset({0, 1})).relaxation_lineaire()
    solution = resoudre(relaxation)
    assert solution.point == (R(25, 6), R(4, 3))
    assert solution.valeur == R(-91, 6)


def test_le_domaine_du_probleme_entier_a_onze_points() -> None:
    """Page 290, figure XIV.2."""
    assert len(points_entiers(_lineaire(), BOITE)) == 11


def test_l_optimum_entier_est_en_3_2() -> None:
    """Page 290 : `z_ILP = -13`. Verifie ici par enumeration des onze points."""
    programme = _lineaire()
    meilleur = min(points_entiers(programme, BOITE), key=programme.valeur)
    assert meilleur == (3, 2)
    assert programme.valeur(meilleur) == -13


def test_un_point_admissible_respecte_contraintes_signes_et_integrite() -> None:
    mixte = ProgrammeEntier(_lineaire(), frozenset({0}))
    assert mixte.est_admissible((4, R(3, 2)))
    assert not mixte.est_admissible((R(25, 6), R(4, 3)))
    assert not mixte.est_admissible((5, 0))
    assert not mixte.est_admissible((1, -1))
    assert not ProgrammeEntier(_lineaire(), frozenset({0, 1})).est_admissible(
        (4, R(3, 2))
    )


def test_un_polyedre_plus_restrictif_donne_une_meilleure_borne() -> None:
    """Page 293 : `D2 = D1` et `x1 <= 7/2` ; `(7/2, 2)`, `-91/6 <= -29/2 <= -13`."""
    d2 = _lineaire(Contrainte((1, 0), INF, R(7, 2)))
    assert points_entiers(d2, BOITE) == points_entiers(_lineaire(), BOITE)
    solution = resoudre(d2)
    assert solution.point == (R(7, 2), 2)
    assert solution.valeur == R(-29, 2)
    assert resoudre(_lineaire()).valeur <= solution.valeur <= -13


def test_la_fermeture_convexe_decrit_le_meme_ensemble_de_points() -> None:
    """Page 294 : `S = D n Z^n` pour les deux polyedres."""
    assert points_entiers(_fermeture_convexe(), BOITE) == points_entiers(
        _lineaire(), BOITE
    )


def test_les_sommets_de_la_fermeture_convexe_sont_entiers() -> None:
    """Page 294 : « les sommets du polyedre sont des points de S »."""
    for sommet in sommets(forme_standard(_fermeture_convexe()), 2):
        assert all(coordonnee.is_integer for coordonnee in sommet)


def test_sur_la_fermeture_convexe_le_simplexe_rend_l_optimum_entier() -> None:
    """Page 294 : `z_LP = z_ILP`, la « facon ideale » de resoudre (ILP)."""
    solution = resoudre(_fermeture_convexe())
    assert solution.point == (3, 2)
    assert solution.valeur == -13


def test_la_matrice_d_un_probleme_de_transport_est_totalement_unimodulaire() -> None:
    """XIV.1.4 : deux `1` par colonne, un par groupe de lignes."""
    matrice = sp.Matrix([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    assert est_totalement_unimodulaire(matrice)


def test_un_coefficient_hors_de_0_1_moins_1_exclut_l_unimodularite() -> None:
    """Page 295 : « les coefficients ne peuvent valoir que 0, 1 ou -1 »."""
    assert not est_totalement_unimodulaire(sp.Matrix([[1, 2], [0, 1]]))


def test_un_mineur_d_ordre_deux_peut_suffire_a_l_exclure() -> None:
    """`[[1, 1], [-1, 1]]` : coefficients permis, determinant 2."""
    assert not est_totalement_unimodulaire(sp.Matrix([[1, 1], [-1, 1]]))


def test_un_mineur_d_ordre_trois_peut_suffire_a_l_exclure() -> None:
    """Le cycle impair : mineurs d'ordre deux dans {0, 1, -1}, determinant 2."""
    assert not est_totalement_unimodulaire(sp.Matrix([[1, 1, 0], [0, 1, 1], [1, 0, 1]]))


def test_une_matrice_totalement_unimodulaire_a_des_sommets_entiers() -> None:
    """Propriete de totale unimodularite, page 295, sur `T x <= d`, `d` entier."""
    matrice = sp.Matrix([[1, 1, 0], [0, 1, 1]])
    assert est_totalement_unimodulaire(matrice)
    contraintes = tuple(
        Contrainte(tuple(matrice[i, :]), INF, d) for i, d in enumerate((3, 5))
    )
    programme = ProgrammeGeneral(Optimisation.MIN, (-1, -1, -1), contraintes)
    for sommet in sommets(forme_standard(programme), 3):
        assert all(coordonnee.is_integer for coordonnee in sommet)


def test_la_boite_d_enumeration_inclut_ses_deux_bords() -> None:
    """`(3, 2)` est sur le bord haut de la boite `[0, 3] x [0, 2]`."""
    assert (3, 2) in points_entiers(_lineaire(), ((0, 3), (0, 2)))
    assert (0, 0) in points_entiers(_lineaire(), ((0, 3), (0, 2)))


def test_une_variable_de_signe_quelconque_peut_etre_negative() -> None:
    """La non-negativite ne s'impose qu'aux variables qui ne sont pas libres."""
    libre = ProgrammeGeneral(
        Optimisation.MIN, (1, 1), (Contrainte((1, 1), INF, 3),), libres=frozenset({1})
    )
    entier = ProgrammeEntier(libre, frozenset({0, 1}))
    assert entier.est_admissible((1, -2))
    assert not entier.est_admissible((-1, 2))
