"""Chapitre I.3.1 a I.3.3 d'optimisation : les rappels d'algebre lineaire.

Oracles ecrits a la main, sauf un, tire du cours : la remarque 3 de la
page 45 dit que les variables de base sont les coordonnees de `d` dans
la base `{t_i}` -- et le tableau de la page 44 donne ces coordonnees
pour la base `{1, 2, 4}` : `(5, 3, 7)`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.algebre_lineaire import (
    NatureDuSysteme,
    coordonnees_dans_la_base,
    est_reguliere,
    nature_du_systeme,
    rang,
    sont_lineairement_independants,
    substituer_dans_la_base,
    systeme_equivalent,
)

T = sp.Matrix([[1, 1, 1, 0, 0], [-2, 3, 0, 1, 0], [1, -1, 0, 0, 1]])
D = sp.Matrix([8, 6, 2])


def _colonnes(*indices: int) -> list[sp.Matrix]:
    return [T[:, i - 1] for i in indices]


def test_deux_vecteurs_non_colineaires_sont_independants() -> None:
    assert sont_lineairement_independants([sp.Matrix([1, 0]), sp.Matrix([1, 1])])


def test_deux_vecteurs_colineaires_sont_lies() -> None:
    assert not sont_lineairement_independants([sp.Matrix([1, 2]), sp.Matrix([2, 4])])


def test_trois_vecteurs_du_plan_sont_lies() -> None:
    """La dimension borne le nombre de vecteurs independants."""
    vecteurs = [sp.Matrix([1, 0]), sp.Matrix([0, 1]), sp.Matrix([1, 1])]
    assert not sont_lineairement_independants(vecteurs)


def test_le_rang_compte_les_lignes_comme_les_colonnes() -> None:
    """Le nombre maximal de lignes independantes egale celui des colonnes."""
    matrice = sp.Matrix([[1, 2, 3], [2, 4, 6], [1, 0, 1]])
    assert rang(matrice) == rang(matrice.T) == 2


def test_une_matrice_carree_de_rang_plein_est_reguliere() -> None:
    assert est_reguliere(sp.Matrix([[1, 1], [1, -1]]))
    assert not est_reguliere(sp.Matrix([[1, 2], [2, 4]]))


def test_une_matrice_non_carree_n_est_pas_reguliere() -> None:
    assert not est_reguliere(sp.Matrix([[1, 0, 0], [0, 1, 0]]))


def test_les_coordonnees_de_d_sont_les_variables_de_base() -> None:
    """Remarque 3, page 45 : `d = somme(x_i t_i)` sur la base `{1, 2, 4}`."""
    assert list(coordonnees_dans_la_base(D, _colonnes(1, 2, 4))) == [5, 3, 7]


def test_des_vecteurs_lies_ne_donnent_pas_de_coordonnees() -> None:
    with pytest.raises(ValueError, match="ne forment pas une base"):
        coordonnees_dans_la_base(
            sp.Matrix([1, 1]), [sp.Matrix([1, 2]), sp.Matrix([2, 4])]
        )


def test_substituer_un_vecteur_de_coordonnee_non_nulle_donne_une_base() -> None:
    """`x = 3 e1 + 2 e2` peut remplacer `e1` : sa coordonnee y vaut 3."""
    base = [sp.Matrix([1, 0]), sp.Matrix([0, 1])]
    nouvelle = substituer_dans_la_base(base, 0, sp.Matrix([3, 2]))
    assert nouvelle == [sp.Matrix([3, 2]), sp.Matrix([0, 1])]
    assert sont_lineairement_independants(nouvelle)


def test_substituer_un_vecteur_de_coordonnee_nulle_est_refuse() -> None:
    """`x = e1` ne peut pas remplacer `e2` : il resterait deux fois `e1`."""
    base = [sp.Matrix([1, 0]), sp.Matrix([0, 1])]
    with pytest.raises(ValueError, match="coordonnee nulle"):
        substituer_dans_la_base(base, 1, sp.Matrix([1, 0]))


def test_un_systeme_est_impossible_si_le_second_membre_ajoute_du_rang() -> None:
    nature = nature_du_systeme(sp.Matrix([[1, 1], [1, 1]]), sp.Matrix([1, 2]))
    assert nature is NatureDuSysteme.IMPOSSIBLE


def test_un_systeme_de_rang_n_a_une_solution_unique() -> None:
    nature = nature_du_systeme(sp.Matrix([[1, 1], [1, -1]]), sp.Matrix([2, 0]))
    assert nature is NatureDuSysteme.SOLUTION_UNIQUE


def test_un_systeme_de_rang_inferieur_a_n_a_une_infinite_de_solutions() -> None:
    assert nature_du_systeme(T, D) is NatureDuSysteme.INFINITE_DE_SOLUTIONS


def test_multiplier_par_une_matrice_reguliere_garde_les_solutions() -> None:
    """`{x | A x = b} = {x | B A x = B b}`."""
    matrice, second = sp.Matrix([[1, 1], [1, -1]]), sp.Matrix([2, 0])
    reguliere = sp.Matrix([[2, 1], [0, 3]])
    produit, nouveau = systeme_equivalent(matrice, second, reguliere)
    assert produit.solve(nouveau) == matrice.solve(second) == sp.Matrix([1, 1])


def test_multiplier_par_une_matrice_singuliere_est_refuse() -> None:
    with pytest.raises(ValueError, match="matrice non reguliere"):
        systeme_equivalent(sp.eye(2), sp.Matrix([1, 1]), sp.Matrix([[1, 2], [2, 4]]))
