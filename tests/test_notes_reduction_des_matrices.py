"""Notes manuscrites d'algebre lineaire, annexe 1 : reduire une matrice.

Source : `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, pages 22 a
31 du fichier (un scan manuscrit, lu en image). Quatre exercices sur les
valeurs propres : deux matrices diagonalisables, deux qui ne le sont pas
et que l'on rend triangulaires en completant les vecteurs propres en une
base. Et la discussion d'un systeme a parametre (« Algebre 2 », page 31).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.discussion import discuter
from polytech.maths.domain.algebre.jordan import multiplicites
from polytech.maths.domain.algebre.methodes import par_cramer
from polytech.maths.domain.algebre.solutions import Nature, solution_generale
from polytech.maths.domain.algebre.spectre import diagonalisee_par
from polytech.maths.domain.algebre.systemes import SystemeLineaire
from polytech.maths.domain.algebre.triangularisation import (
    completer_en_base,
    est_triangulaire_superieure,
    est_triangularisable,
    produit_de_passage,
    reconstruit_par,
    semblable_par,
    semblable_par_inverse_donnee,
    triangulariser,
)
from polytech.maths.domain.comparaison import egales

R = sp.Rational
T = sp.Symbol("t")
EXERCICE_2 = sp.Matrix([[2, 0, 1, 1], [-2, 2, 1, -1], [0, 0, 3, 1], [0, -2, -1, 3]])
EXERCICE_3 = sp.Matrix([[-15, 1, -9], [0, 6, 0], [4, 1, -3]])
EXERCICE_4 = sp.Matrix([[1, -3, 3], [3, -5, 3], [6, -6, 4]])
A = sp.Symbol("a", real=True)


def test_l_exercice_1_se_diagonalise() -> None:
    """`[[2, 2], [1, 3]]` : `P = [[-2, 1], [1, 1]]`, `D = diag(1, 4)`."""
    matrice, passage = sp.Matrix([[2, 2], [1, 3]]), sp.Matrix([[-2, 1], [1, 1]])
    assert passage.inv() == -sp.Matrix([[1, -1], [-1, -2]]) / 3
    assert diagonalisee_par(matrice, passage) == sp.diag(1, 4)
    assert multiplicites(matrice) == {1: (1, 1), 4: (1, 1)}


def test_le_polynome_caracteristique_de_l_exercice_2() -> None:
    """`det(A - tI) = (t - 2)^3 (t - 4)` ; un seul vecteur propre par valeur propre."""
    caracteristique = (EXERCICE_2 - T * sp.eye(4)).det()
    assert sp.expand(caracteristique - (T - 4) * (T - 2) ** 3) == 0
    assert multiplicites(EXERCICE_2) == {2: (3, 1), 4: (1, 1)}
    assert EXERCICE_2 * sp.Matrix([-1, 1, -1, 1]) == 2 * sp.Matrix([-1, 1, -1, 1])
    assert EXERCICE_2 * sp.Matrix([1, -1, 1, 1]) == 4 * sp.Matrix([1, -1, 1, 1])


def test_la_premiere_etape_de_la_reduction_de_l_exercice_2() -> None:
    """`P_1` complete les deux vecteurs propres ; reste `A_1 = [[2, 0], [-2, 2]]`."""
    premiere = sp.Matrix([[-1, 1, 1, 0], [1, -1, 0, 1], [-1, 1, 0, 0], [1, 1, 0, 0]])
    attendue = sp.Matrix([[2, 0, 0, -1], [0, 4, 0, -1], [0, 0, 2, 0], [0, 0, -2, 2]])
    assert semblable_par(EXERCICE_2, premiere) == attendue
    assert attendue[2:, 2:] == sp.Matrix([[2, 0], [-2, 2]])
    assert attendue[2:, :2] == sp.zeros(2, 2)


def test_la_matrice_triangulaire_de_l_exercice_2() -> None:
    """`P^-1 A P = [[2, 0, -1, 0], [0, 4, -1, 0], [0, 0, 2, -2], [0, 0, 0, 2]]`."""
    passage = sp.Matrix([[-1, 1, 0, 1], [1, -1, 1, 0], [-1, 1, 0, 0], [1, 1, 0, 0]])
    attendue = sp.Matrix([[2, 0, -1, 0], [0, 4, -1, 0], [0, 0, 2, -2], [0, 0, 0, 2]])
    assert semblable_par(EXERCICE_2, passage) == attendue
    assert est_triangulaire_superieure(attendue)


def test_l_exercice_3_n_est_pas_diagonalisable() -> None:
    """`(6 - t)(t + 9)^2` ; pour `-9`, `m_g = 1 < m_a = 2`."""
    caracteristique = (EXERCICE_3 - T * sp.eye(3)).det()
    assert sp.expand(caracteristique + (T - 6) * (T + 9) ** 2) == 0
    assert multiplicites(EXERCICE_3) == {-9: (2, 1), 6: (1, 1)}
    assert not EXERCICE_3.is_diagonalizable()
    assert est_triangularisable(EXERCICE_3)


def test_la_base_completee_de_l_exercice_3() -> None:
    """Les vecteurs propres `(0, 9, 1)` et `(-3, 0, 2)`, prolonges par `(1, 0, 0)`."""
    propres = [sp.Matrix([0, 9, 1]), sp.Matrix([-3, 0, 2])]
    passage = completer_en_base(propres)
    assert passage == sp.Matrix([[0, -3, 1], [9, 0, 0], [1, 2, 0]])
    assert passage.inv() == sp.Matrix([[0, 2, 0], [0, -1, 9], [18, -3, 27]]) / 18


def test_la_matrice_triangulaire_de_l_exercice_3() -> None:
    """`P_1^-1 A P_1 = [[6, 0, 0], [0, -9, 2], [0, 0, -9]]`, « en une seule etape »."""
    passage = sp.Matrix([[0, -3, 1], [9, 0, 0], [1, 2, 0]])
    attendue = sp.Matrix([[6, 0, 0], [0, -9, 2], [0, 0, -9]])
    assert semblable_par(EXERCICE_3, passage) == attendue


@pytest.mark.parametrize("matrice", [EXERCICE_2, EXERCICE_3, EXERCICE_4])
def test_la_triangularisation_rend_une_matrice_semblable_et_triangulaire(
    matrice: sp.Matrix,
) -> None:
    """`P^-1 A P = T`, triangulaire, les valeurs propres sur la diagonale."""
    passage, triangulaire = triangulariser(matrice)
    assert passage.inv() * matrice * passage == triangulaire
    assert est_triangulaire_superieure(triangulaire)
    diagonale = sorted(triangulaire.diagonal())
    attendue = sorted(
        v for v, (m, _) in multiplicites(matrice).items() for _ in range(m)
    )
    assert diagonale == attendue


def test_l_exercice_4_se_diagonalise() -> None:
    """`(t + 2)^2 (4 - t)`, `m_a = m_g` : `P^-1 A P = diag(-2, -2, 4)`."""
    assert multiplicites(EXERCICE_4) == {-2: (2, 2), 4: (1, 1)}
    passage = sp.Matrix([[1, -1, 1], [1, 0, 1], [0, 1, 2]])
    assert passage.inv() == sp.Matrix([[-1, 3, -1], [-2, 2, 0], [1, -1, 1]]) / 2
    assert diagonalisee_par(EXERCICE_4, passage) == sp.diag(-2, -2, 4)
    _, triangulaire = triangulariser(EXERCICE_4)
    assert triangulaire.is_diagonal()


def test_une_rotation_n_est_pas_triangularisable_sur_les_reels() -> None:
    """Ses valeurs propres sont `+-j` : la somme des `m_a` reels ne fait pas `n`."""
    rotation = sp.Matrix([[0, -1], [1, 0]])
    assert not est_triangularisable(rotation)
    with pytest.raises(ValueError, match="non triangularisable sur les reels"):
        triangulariser(rotation)


def test_une_famille_liee_ne_se_complete_pas_en_base() -> None:
    with pytest.raises(ValueError, match="famille liee"):
        completer_en_base([sp.Matrix([1, 2]), sp.Matrix([2, 4])])


def test_la_discussion_du_systeme_a_parametre() -> None:
    """`ax - 6y = 5a - 3`, `2x + (a - 7)y = 29 - 7a` : `D = a^2 - 7a + 12`."""
    systeme = SystemeLineaire(
        sp.Matrix([[A, -6], [2, A - 7]]), sp.Matrix([5 * A - 3, 29 - 7 * A])
    )
    assert sp.expand(systeme.matrice.det()) == A**2 - 7 * A + 12
    assert discuter(systeme, A) == {3: Nature.INDETERMINE, 4: Nature.IMPOSSIBLE}


def test_la_solution_hors_des_valeurs_singulieres() -> None:
    """`x = (5a^2 - 80a + 195)/D`, `y = (-7a^2 + 19a + 6)/D`."""
    systeme = SystemeLineaire(
        sp.Matrix([[A, -6], [2, A - 7]]), sp.Matrix([5 * A - 3, 29 - 7 * A])
    )
    d = A**2 - 7 * A + 12
    attendue = sp.Matrix([(5 * A**2 - 80 * A + 195) / d, (-7 * A**2 + 19 * A + 6) / d])
    assert egales(par_cramer(systeme), attendue)


def test_pour_a_egal_trois_le_systeme_est_indetermine() -> None:
    """`3x - 6y = 12` et `2x - 4y = 8` : `x = 2y + 4`, avec `y` pour parametre."""
    systeme = SystemeLineaire(sp.Matrix([[3, -6], [2, -4]]), sp.Matrix([12, 8]))
    x, y = solution_generale(systeme)
    assert egales(x, 2 * y + 4)


def test_la_solution_ecrite_page_33_a_perdu_un_facteur() -> None:
    """Le manuscrit ecrit `x = (6/a) y + a - 3/a = 2y + 2` ; c'est `(5a - 3)/a`, soit 4.

    Scan relu a plus haute resolution. Toute la chaine passe par la
    bibliotheque : la forme generale en `a` (premiere equation resolue en
    `x`), sa valeur en `a = 3` (le systeme indetermine), puis le residu de la
    seconde equation `2x - 4y = 8` pour chacune des deux formes.
    """
    premiere = SystemeLineaire(sp.Matrix([[A, -6]]), sp.Matrix([5 * A - 3]))
    x_general, libre = solution_generale(premiere)
    assert egales(x_general, 6 * libre / A + (5 * A - 3) / A)
    assert not egales(x_general, 6 * libre / A + A - 3 / A)

    systeme = SystemeLineaire(sp.Matrix([[3, -6], [2, -4]]), sp.Matrix([12, 8]))
    x_lib, y_lib = solution_generale(systeme)
    assert egales(x_general.subs(A, 3), x_lib.subs(y_lib, libre))
    assert sp.expand(x_lib - (2 * y_lib + 4)) == 0

    def residu(x: sp.Expr, y: sp.Expr) -> sp.Matrix:
        return systeme.matrice * sp.Matrix([x, y]) - systeme.second_membre

    ecrite = (6 * libre / A + A - 3 / A).subs(A, 3)
    assert sp.expand(ecrite - (2 * libre + 2)) == 0
    assert sp.expand(residu(x_general.subs(A, 3), libre)) == sp.zeros(2, 1)
    assert sp.expand(residu(ecrite, libre)) == sp.Matrix([-6, -4])


def test_un_inverse_donne_qui_n_en_est_pas_un_leve() -> None:
    passage = sp.Matrix([[1, 0], [0, 2]])
    with pytest.raises(ValueError, match="pas l'inverse"):
        semblable_par_inverse_donnee(sp.eye(2), passage, sp.eye(2))
    assert produit_de_passage(passage, sp.eye(2)) == passage
    inverse = sp.Matrix([[1, 0], [0, R(1, 2)]])
    assert semblable_par_inverse_donnee(sp.eye(2), passage, inverse) == sp.eye(2)
    assert reconstruit_par(passage, sp.eye(2), inverse) == sp.eye(2)
