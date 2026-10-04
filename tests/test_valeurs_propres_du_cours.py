"""Mathematique, planches 20 a 28 : valeurs propres, diagonalisation, espace vectoriel.

Oracles (`Mathematique/SLIDES/Slides_01_Algebre_lineaire_et_integrales.pdf`) :
`A = [[2, 2], [1, 3]]` -- equation aux valeurs propres
`lambda^2 - 5 lambda + 4 = 0`, valeurs propres 1 et 4, vecteurs propres
`k (-2, 1)` et `k (1, 1)` (planches 21 a 24) ; `P = [[-2, 1], [1, 1]]`,
`A P = [[-2, 4], [1, 4]] = P D`, `P^-1 A P = D` (planches 25 a 27).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.espaces import (
    combinaison_lineaire,
    sont_colineaires,
)
from polytech.maths.domain.algebre.spectre import INCONNUE, Spectre, diagonalisee_par

A = sp.Matrix([[2, 2], [1, 3]])
P = sp.Matrix([[-2, 1], [1, 1]])


def test_l_equation_aux_valeurs_propres_de_la_planche_23() -> None:
    """`(2 - lambda)(3 - lambda) - 2 = lambda^2 - 5 lambda + 4`."""
    attendu = INCONNUE**2 - 5 * INCONNUE + 4
    assert Spectre(A).polynome_caracteristique() == attendu
    assert Spectre(A).valeurs_propres() == (1, 4)


def test_une_valeur_propre_rend_le_systeme_homogene_indetermine() -> None:
    """Planche 22 : `rang(A - lambda I) < 2` exactement pour les valeurs propres."""
    assert (A - 1 * sp.eye(2)).rank() == 1
    assert (A - 4 * sp.eye(2)).rank() == 1
    assert (A - 2 * sp.eye(2)).rank() == 2


def test_les_vecteurs_propres_de_la_planche_24() -> None:
    """`k (-2, 1)` pour 1, `k (1, 1)` pour 4, `k` non nul."""
    couples = dict(Spectre(A).couples_propres())
    assert sont_colineaires(couples[1], sp.Matrix([-2, 1]))
    assert sont_colineaires(couples[4], sp.Matrix([1, 1]))
    assert not sont_colineaires(couples[1], couples[4])


def test_un_vecteur_propre_garde_sa_direction() -> None:
    """Planche 20 : `A x` est un multiple de `x`."""
    assert A * sp.Matrix([-2, 1]) == 1 * sp.Matrix([-2, 1])
    assert A * sp.Matrix([1, 1]) == 4 * sp.Matrix([1, 1])
    assert not sont_colineaires(A * sp.Matrix([1, 0]), sp.Matrix([1, 0]))


def test_le_vecteur_nul_n_est_colineaire_a_rien_au_sens_des_vecteurs_propres() -> None:
    """Planche 20 : `x != 0` ; le vecteur nul n'est pas un vecteur propre."""
    assert not sont_colineaires(sp.Matrix([0, 0]), sp.Matrix([1, 1]))


def test_le_produit_ap_de_la_planche_26() -> None:
    """`A P = [[-2, 4], [1, 4]] = P diag(1, 4)`."""
    produit = A * P
    assert produit == sp.Matrix([[-2, 4], [1, 4]])
    assert produit == P * sp.diag(1, 4)


def test_la_matrice_des_vecteurs_propres_diagonalise() -> None:
    """Planche 27 : `P^-1 A P = D`."""
    assert diagonalisee_par(A, P) == sp.diag(1, 4)


def test_une_matrice_de_passage_quelconque_ne_diagonalise_pas() -> None:
    with pytest.raises(ValueError, match="ne sont pas des vecteurs propres"):
        diagonalisee_par(A, sp.Matrix([[1, 0], [1, 1]]))


def test_des_colonnes_liees_ne_forment_pas_une_matrice_de_passage() -> None:
    """Planche 25 : il faut une **base** de vecteurs propres."""
    with pytest.raises(ValueError, match="ne forment pas une base"):
        diagonalisee_par(A, sp.Matrix([[1, 2], [1, 2]]))


def test_un_espace_vectoriel_est_stable_par_combinaison_lineaire() -> None:
    """Planche 28 : `k x` et `x + y` restent dans `R^n`."""
    x, y = sp.Matrix([1, 2, 3]), sp.Matrix([0, -1, 4])
    assert combinaison_lineaire((2, -1), (x, y)) == sp.Matrix([2, 5, 2])
    assert combinaison_lineaire((1, 1), (x, y)) == x + y
    assert combinaison_lineaire((5,), (x,)) == 5 * x


def test_une_combinaison_demande_autant_de_coefficients_que_de_vecteurs() -> None:
    with pytest.raises(ValueError, match="coefficients pour"):
        combinaison_lineaire((1, 2), (sp.Matrix([1, 0]),))


def test_des_vecteurs_de_dimensions_differentes_ne_se_combinent_pas() -> None:
    with pytest.raises(ValueError, match="meme espace"):
        combinaison_lineaire((1, 1), (sp.Matrix([1, 0]), sp.Matrix([1, 0, 0])))
