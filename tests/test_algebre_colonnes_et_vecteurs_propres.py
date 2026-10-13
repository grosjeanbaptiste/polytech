"""Syllabus d'algebre : les colonnes d'un determinant, et le vecteur propre.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`.

- V.1 (page 55) : les operations elementaires se font « sur les lignes
  et/ou sur les colonnes », et les trois proprietes valent pour une
  « rangee » ;
- X.2.4 (pages 157 et 158) : le polynome caracteristique de l'exemple se
  calcule par `c1 <- c1 + c2`, `l2 <- l2 - l1`, `c1 <- c1 + c3`,
  `l3 <- l3 - l1`, et les trois vecteurs propres sont imprimes ;
- X.1.2, exercice 3 (page 155) : « le vecteur (1, 1, 1) est-il un vecteur
  propre ? Si oui, quelle est la valeur propre associee ? » ;
- II.6 (page 21) : une matrice triangulaire est une matrice carree.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre import operations_elementaires as lignes
from polytech.maths.domain.algebre.operations_sur_les_colonnes import (
    combiner,
    multiplier,
    permuter,
)
from polytech.maths.domain.algebre.spectre import valeur_propre_associee
from polytech.maths.domain.algebre.triangularisation import (
    est_triangulaire_superieure,
)

T = sp.Symbol("t")
A = sp.Matrix([[3, 1, -3], [3, 2, -4], [5, -1, -3]])
"""La matrice de l'exemple X.2.4, page 157."""


def test_combiner_deux_colonnes_laisse_le_determinant() -> None:
    obtenue = combiner(A - T * sp.eye(3), 0, 1, 1)
    attendue = sp.Matrix([[4 - T, 1, -3], [5 - T, 2 - T, -4], [4, -1, -3 - T]])
    assert obtenue.matrice == attendue
    assert obtenue.facteur == 1


def test_le_polynome_caracteristique_de_la_page_157_par_ses_quatre_gestes() -> None:
    etape = combiner(A - T * sp.eye(3), 0, 1, 1).matrice
    etape = lignes.combiner(etape, 1, -1, 0).matrice
    etape = combiner(etape, 0, 1, 2).matrice
    etape = lignes.combiner(etape, 2, -1, 0).matrice
    assert etape == sp.Matrix([[1 - T, 1, -3], [0, 1 - T, -1], [0, -2, -T]])
    attendu = -(T - 1) * (T + 1) * (T - 2)
    assert sp.expand(etape.det() - attendu) == 0
    assert sp.expand((A - T * sp.eye(3)).det() - attendu) == 0


def test_permuter_deux_colonnes_change_le_signe() -> None:
    obtenue = permuter(A, 0, 2)
    assert obtenue.matrice == sp.Matrix([[-3, 1, 3], [-4, 2, 3], [-3, -1, 5]])
    assert obtenue.facteur == -1
    assert obtenue.matrice.det() == obtenue.facteur * A.det()


def test_multiplier_une_colonne_multiplie_le_determinant() -> None:
    obtenue = multiplier(A, 1, sp.Rational(1, 2))
    assert obtenue.matrice[:, 1] == sp.Matrix(
        [sp.Rational(1, 2), 1, -sp.Rational(1, 2)]
    )
    assert obtenue.matrice.det() == obtenue.facteur * A.det()


def test_une_colonne_ne_se_combine_pas_avec_elle_meme() -> None:
    with pytest.raises(ValueError, match="colonne ne se combine pas avec elle-meme"):
        combiner(A, 1, 2, 1)


@pytest.mark.parametrize(
    ("vecteur", "valeur"),
    [((1, 1, 1), 1), ((5, 7, 9), -1), ((4, 5, 3), 2)],
)
def test_les_trois_vecteurs_propres_de_la_page_158(
    vecteur: tuple[int, ...], valeur: int
) -> None:
    assert valeur_propre_associee(A, sp.Matrix(vecteur)) == valeur


def test_l_exercice_3_de_la_page_155() -> None:
    matrice = sp.Matrix([[1, 2, 3], [3, 0, 3], [1, 4, 1]])
    assert valeur_propre_associee(matrice, sp.Matrix([1, 1, 1])) == 6


def test_un_vecteur_qui_n_est_pas_propre_leve() -> None:
    with pytest.raises(ValueError, match="n'est pas un vecteur propre"):
        valeur_propre_associee(A, sp.Matrix([1, 0, 0]))


def test_le_vecteur_nul_n_est_jamais_propre() -> None:
    with pytest.raises(ValueError, match="vecteur nul"):
        valeur_propre_associee(A, sp.zeros(3, 1))


def test_un_vecteur_d_une_autre_taille_leve() -> None:
    with pytest.raises(ValueError, match="2 composantes pour une matrice d'ordre 3"):
        valeur_propre_associee(A, sp.Matrix([1, 1]))


def test_une_matrice_rectangulaire_n_est_pas_triangulaire() -> None:
    assert not est_triangulaire_superieure(sp.Matrix([[1, 2, 3], [0, 4, 5]]))


def test_une_matrice_carree_nulle_sous_sa_diagonale_est_triangulaire() -> None:
    assert est_triangulaire_superieure(sp.Matrix([[1, 2], [0, 4]]))
