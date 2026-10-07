"""Optimisation : les exemples resolus, partie I (l'algorithme simplexe).

Source : `Optimisation/EXS/Exs_Exemples_resolus.pdf`, planches 3 a 44. Les
planches illustrent le cours : formes d'un programme, sommets, changement
de base, puis six deroulements du simplexe. Chaque resultat imprime est
recopie comme oracle et recalcule ; tous sont retrouves.
"""

from __future__ import annotations

import sympy as sp

from polytech.optimisation.application import exemples_formes as fo
from polytech.optimisation.application import exemples_issues as iss
from polytech.optimisation.application import exemples_sommets as so
from polytech.optimisation.domain.algorithme import Issue

Q = sp.Rational
M = iss.M

# --- Planches 3 a 7 : les formes d'un programme -----------------------------


def test_les_deux_ecritures_des_variables_libres() -> None:
    """Planches 4 et 5 : cinq variables avec `x+ - x-`, quatre avec la translation."""
    assert fo.variables(decomposees=True) == 5
    assert fo.variables(decomposees=False) == 4


def test_la_forme_translatee_de_la_planche_5() -> None:
    """Planche 5 : `min -x1' + 2x2 - 4x3' + 5 xbar`, trois contraintes."""
    objectif, lignes = fo.translatee()
    assert objectif == (-1, 2, -4, 5)
    assert lignes == (
        ((1, 3, 2, -3), "<=", 10),
        ((2, -1, 6, -8), ">=", 5),
        ((1, 2, 3, -4), "=", 6),
    )


def test_la_forme_canonique_de_la_planche_6() -> None:
    """Planche 6 : quatre contraintes `>=`, l'egalite etant dedoublee."""
    assert fo.canonique() == (
        ((-1, -3, -2, 3), -10),
        ((2, -1, 6, -8), 5),
        ((1, 2, 3, -4), 6),
        ((-1, -2, -3, 4), -6),
    )


def test_le_probleme_des_planches_3_a_7_n_est_pas_borne() -> None:
    """Les planches ne le resolvent pas : son objectif croit sans fin."""
    assert fo.issue() is Issue.NON_BORNEE
    for t in (0, 10, 1000):
        assert fo.est_admissible(fo.point_du_rayon(t))
        assert fo.valeur(fo.point_du_rayon(t)) == 6 + t


# --- Planches 8 a 13 : les sommets ------------------------------------------


def test_les_dix_bases_de_la_planche_9() -> None:
    """Planche 9 : les dix matrices 3 x 3 issues de `T` sont regulieres."""
    assert so.bases_regulieres(so.PREMIER) == 10


def test_un_sommet_et_un_point_non_admissible() -> None:
    """Planches 10 et 11 : `(5, 3)` avec `x2e = 7` ; puis `x2 = -2`."""
    assert so.solution_de_base(so.PREMIER, (1, 2, 4)) == {1: 5, 2: 3, 4: 7}
    assert so.solution_de_base(so.PREMIER, (2, 3, 4)) == {2: -2, 3: 10, 4: 12}
    assert so.est_un_sommet(so.PREMIER, (1, 2, 4))
    assert not so.est_un_sommet(so.PREMIER, (2, 3, 4))


def test_les_deux_ensembles_qui_ne_sont_pas_des_bases() -> None:
    """Planche 13 : `{1, 3, 5}` et `{2, 4, 5}` ne forment pas des bases."""
    assert so.ensembles_singuliers(so.SECOND) == ((1, 3, 5), (2, 4, 5))


# --- Planches 24 a 30 : l'exemple I -----------------------------------------


def test_exemple_i_les_trois_tableaux() -> None:
    """Planches 27 a 29 : `X0 = (0,0)`, `X1 = (2,0)`, `X2 = (5,3)` ; z = 0, -12, -45."""
    assert so.parcours() == (((0, 0), 0), ((2, 0), -12), ((5, 3), -45))


def test_exemple_i_la_premiere_ligne_du_tableau_optimal() -> None:
    """Planche 29 : les couts reduits valent -1/2 et -11/2."""
    assert so.couts_reduits_finaux() == {5: Q(-1, 2), 3: Q(-11, 2)}


def test_exemple_i_l_optimum() -> None:
    """Second chemin : le maximum de `6x1 + 5x2` vaut 45, en `(5, 3)`."""
    assert iss.optimum("I") == (45, (5, 3))


# --- Planches 31 a 40 : les trois issues ------------------------------------


def test_exemple_ii_non_borne() -> None:
    """Planche 32 : aucun pivot positif possible, `z = -infini`."""
    assert iss.issue("II") is Issue.NON_BORNEE


def test_exemple_i_avec_une_contrainte_retournee() -> None:
    """Planche 36 : `z = 122` en `(12, 10)`."""
    assert iss.optimum("I retourne") == (122, (12, 10))


def test_exemple_ii_impossible() -> None:
    """Planche 40 : `Pa` est optimal avec `x2a = 14/5` : pas de solution pour `P`."""
    assert iss.issue("II retourne") is Issue.IMPOSSIBLE
    valeur, artificielle = iss.optimum_du_probleme_penalise()
    assert sp.expand(valeur - (Q(14, 5) * M - Q(218, 5))) == 0
    assert artificielle == Q(14, 5)


# --- Planches 41 a 44 : degenerescence, solutions multiples -----------------


def test_trois_bases_pour_un_meme_sommet() -> None:
    """Planche 42 : trois bases differentes correspondent a la meme solution."""
    assert so.bases_du_sommet(so.DEGENERE, (2, 0)) == ((1, 2, 5), (1, 3, 5), (1, 4, 5))


def test_une_arete_de_solutions_optimales() -> None:
    """Planche 43 : `(5, 3)` et `(18/5, 22/5)`, de valeur -40."""
    assert so.sommets_optimaux(so.ARETE) == (-40, ((Q(18, 5), Q(22, 5)), (5, 3)))


def test_un_rayon_de_solutions_optimales() -> None:
    """Planche 44 : le sommet `(4, 0)`, puis un rayon ou `z` reste a -8."""
    assert iss.optimum("rayon") == (-8, (4, 0))
    for t in (0, 1, 50):
        assert iss.sur_le_rayon(t) == -8
