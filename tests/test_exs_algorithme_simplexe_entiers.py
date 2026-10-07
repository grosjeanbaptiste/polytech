"""Optimisation : les deux exercices de « Branch and Bound » (V et VI).

Source : `Optimisation/EXS/Exs_Algorithme_simplexe.pdf`, planches 46 a 62.
Les planches minimisent l'oppose de l'objectif : leurs evaluations sont
negatives. Chaque evaluation, chaque penalite et chaque optimum est
recopie comme oracle, recalcule, puis recoupe par l'enumeration de tous
les points entiers.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import sympy as sp

from polytech.optimisation.application import simplexe_entiers as en
from polytech.optimisation.domain.algorithme import Issue
from polytech.optimisation.domain.contraintes import Contrainte, Sens

Q = sp.Rational
INF, SUP = Sens.INFERIEUR, Sens.SUPERIEUR

# --- Exercice V -------------------------------------------------------------


def test_v_la_relaxation_lineaire() -> None:
    """Planche 50 : `x1 = 7/2`, `x2 = 5/2`, et `v0 = -23`."""
    noeud = en.noeud(en.CINQ)
    assert noeud.point == (Q(7, 2), Q(5, 2))
    assert noeud.evaluation == -23


def test_v_les_penalites_de_la_planche_50() -> None:
    """Planche 50 : `P1 = (11/4, 1)`, `P2 = (1, +infini)`."""
    assert en.penalites_a_la_racine(en.CINQ) == {1: (Q(11, 4), 1), 2: (1, sp.oo)}


def test_v_la_separation_sur_x1() -> None:
    """Planches 51 et 52 : `x1 <= 3` donne -81/4, `x1 >= 4` donne (4, 2) et -22."""
    gauche = en.noeud(en.CINQ, ((1, INF, 3),))
    droite = en.noeud(en.CINQ, ((1, SUP, 4),))
    assert gauche.evaluation == Q(-81, 4)
    assert (droite.point, droite.evaluation) == ((4, 2), -22)
    assert gauche.evaluation > droite.evaluation


def test_v_la_separation_sur_x2() -> None:
    """Planches 53 et 54 : `x2 <= 2` donne (4, 2) ; `x2 >= 3` est impossible."""
    gauche = en.noeud(en.CINQ, ((2, INF, 2),))
    assert (gauche.point, gauche.evaluation) == ((4, 2), -22)
    assert en.noeud(en.CINQ, ((2, SUP, 3),)).issue is Issue.IMPOSSIBLE


def test_v_l_optimum_entier() -> None:
    """Planche 51 : `x1 = 4`, `x2 = 2` ; l'objectif a maximiser vaut 22."""
    assert en.optimum_entier(en.CINQ) == (22, (4, 2))


def test_v_par_enumeration() -> None:
    """Second chemin : tous les points entiers du domaine, un a un."""
    assert en.meilleur_point_entier(en.CINQ) == (22, ((4, 2),))


# --- Exercice VI ------------------------------------------------------------


def test_vi_la_relaxation_lineaire() -> None:
    """Planche 58 : `x1 = 13/4`, `x2 = 0`, `x3 = 21/4`, et `v0 = -267/4`."""
    noeud = en.noeud(en.SIX)
    assert noeud.point == (Q(13, 4), 0, Q(21, 4))
    assert noeud.evaluation == Q(-267, 4)


def test_vi_les_penalites_de_la_planche_58() -> None:
    """Planche 58 : `P3 = (7/4, 27/4)` et `P1 = (3/4, 63/4)`."""
    attendues = {3: (Q(7, 4), Q(27, 4)), 1: (Q(3, 4), Q(63, 4))}
    assert en.penalites_a_la_racine(en.SIX) == attendues


def test_vi_la_plus_petite_penalite_designe_x1_inferieur_a_3() -> None:
    """Planche 59 : la plus petite penalite est `P1(a)` ; on impose `x1 <= 3`."""
    assert en.plus_petite_penalite(en.SIX) == (1, "a", Q(3, 4))


def test_vi_le_sommet_s1() -> None:
    """Planche 59 : `v1 = -66`, avec `x3 = 16/3` ; il n'est pas entier."""
    noeud = en.noeud(en.SIX, ((1, INF, 3),))
    assert noeud.evaluation == -66
    assert noeud.point == (3, 0, Q(16, 3))


def test_vi_le_sommet_s2() -> None:
    """Planche 60 : `x3 <= 5` donne `x = (3, 1, 5)` et `v2 = -65`, entier."""
    noeud = en.noeud(en.SIX, ((1, INF, 3), (3, INF, 5)))
    assert (noeud.point, noeud.evaluation) == ((3, 1, 5), -65)


def test_vi_les_deux_sommets_sondes_par_leur_borne() -> None:
    """Planche 61 : `v3 >= -60` et `v4 >= -51`, tous deux au-dessus de -65."""
    assert en.borne_par_la_penalite(-66, 6) == -60
    assert en.borne_par_la_penalite(Q(-267, 4), Q(63, 4)) == -51


def test_vi_les_bornes_sont_ici_exactes() -> None:
    """Second chemin : les deux sommets, resolus pour de bon, valent -60 et -51."""
    s3 = en.noeud(en.SIX, ((1, INF, 3), (3, SUP, 6)))
    s4 = en.noeud(en.SIX, ((1, SUP, 4),))
    assert (s3.evaluation, s4.evaluation) == (-60, -51)


def test_vi_l_optimum_entier() -> None:
    """Planche 61 : `x = (3, 1, 5)`, l'objectif a maximiser vaut 65."""
    assert en.optimum_entier(en.SIX) == (65, (3, 1, 5))


def test_vi_par_enumeration() -> None:
    """Second chemin : l'optimum entier est unique."""
    assert en.meilleur_point_entier(en.SIX) == (65, ((3, 1, 5),))


def test_la_boite_a_enumerer() -> None:
    """Chaque variable va de zero au plus petit quotient qui la borne."""
    assert en.boite(en.CINQ) == [(0, 6), (0, 6)]
    assert en.boite(en.SIX) == [(0, 5), (0, 15), (0, 6)]


def test_une_variable_que_rien_ne_borne_est_refusee() -> None:
    """Sans contrainte a coefficients positifs, la boite a enumerer n'a pas de cote."""
    libre = replace(en.CINQ, contraintes=en.CINQ.contraintes[1:])
    with pytest.raises(ValueError, match="aucune contrainte ne borne x") as refus:
        en.meilleur_point_entier(libre)
    assert "x1 a elle seule" in str(refus.value)


def test_un_probleme_sans_point_entier_est_dit_impossible() -> None:
    """`x2 >= 3` vide le domaine de l'exercice V : il n'y a pas d'optimum a rendre."""
    vide = replace(
        en.CINQ, contraintes=(*en.CINQ.contraintes, Contrainte((0, 1), SUP, 3))
    )
    with pytest.raises(ValueError, match="aucun point entier admissible"):
        en.optimum_entier(vide)


def test_une_variable_inconnue_est_refusee() -> None:
    """Une borne sur `x4` n'a pas de sens dans un probleme a trois variables."""
    with pytest.raises(ValueError, match="variable hors du probleme"):
        en.noeud(en.SIX, ((4, INF, 1),))
