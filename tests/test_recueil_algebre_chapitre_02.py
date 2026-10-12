"""Le chapitre II du recueil d'algebre, « Les matrices », refait.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, exercices 1 a 10 des
sections II.5, II.6, II.7.4 et II.8.3 ; l'oracle est la page « Solutions du
2eme chapitre » (pages 29 et 30 du syllabus).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.application import recueil_algebre_ch02_equations as equations
from polytech.maths.application import recueil_algebre_ch02_formes as formes
from polytech.maths.application.macros_recueil_algebre_ch02 import (
    macros_du_chapitre_deux,
)
from polytech.maths.application.registre import JEUX_DE_MACROS
from polytech.maths.domain.algebre.echelonnement import (
    est_echelonnee,
    est_echelonnee_simple,
)
from polytech.maths.domain.symboles import VARIABLE as X

M = sp.Matrix
R = sp.Rational


def test_exercice_1_une_equation_du_premier_degre() -> None:
    attendue = M([[R(29, 3), 2], [R(20, 3), 1], [9, R(5, 3)]])
    assert equations.exercice_1() == attendue


def test_exercice_2_l_inconnue_des_deux_cotes() -> None:
    solution = equations.exercice_2()
    assert solution == M([[-3, 2], [0, -1]])
    a, b, c = equations.DONNEES_2
    assert a * (solution - b) == solution * c


def test_exercice_3_les_matrices_qui_commutent_avec_a() -> None:
    generale = equations.exercice_3()
    assert generale[0, 0] == generale[1, 1] == generale[2, 2]
    assert generale[0, 1] == generale[1, 2]
    assert generale[1, 0] == generale[2, 0] == generale[2, 1] == 0
    assert len(generale.free_symbols) == 3
    assert equations.A_3 * generale == generale * equations.A_3


def test_exercice_4_seules_les_scalaires_commutent_avec_tout() -> None:
    generale = equations.exercice_4()
    assert generale[0, 1] == 0
    assert generale[0, 0] == generale[1, 1]
    autre = M([[1, 2], [3, 4]])
    assert generale * autre == autre * generale


def test_exercice_5_rendre_un_produit_diagonal() -> None:
    hors_diagonale, racines = equations.exercice_5()
    assert sp.expand(hors_diagonale - (-3 * X**2 + 4 * X + 3)) == 0
    assert set(racines) == {(2 + sp.sqrt(13)) / 3, (2 - sp.sqrt(13)) / 3}


def test_exercice_6_parties_symetrique_et_antisymetrique() -> None:
    symetrique, antisymetrique = equations.exercice_6()
    assert symetrique == M([[1, R(5, 2)], [R(5, 2), 4]])
    assert antisymetrique == M([[0, -R(1, 2)], [R(1, 2), 0]])


def test_exercice_7_une_rotation_est_orthogonale() -> None:
    produit = equations.exercice_7()
    assert produit == sp.eye(2)


def test_exercice_8_un_determinant_nul() -> None:
    determinant, racines = equations.exercice_8()
    assert sp.expand(determinant - (X**2 - R(5, 2) * X + 1)) == 0
    assert set(racines) == {sp.Integer(2), R(1, 2)}


@pytest.mark.parametrize(
    ("nom", "attendue"),
    [
        (
            "A",
            [[1, 0, 0, R(15, 7)], [0, 1, 0, -R(4, 7)], [0, 0, 1, -R(10, 7)], [0] * 4],
        ),
        ("B", [[1, 2, 0, 9, 0], [0, 0, 1, -7, 0], [0, 0, 0, 0, 1], [0] * 5]),
        ("C", [[1, 0, R(7, 9)], [0, 1, -R(26, 9)], [0, 0, 0]]),
        ("D", [[1, 0, -5, -5, 0], [0, 1, 2, 1, -1], [0] * 5, [0] * 5]),
    ],
)
def test_exercice_9_formes_echelonnees_simples(nom: str, attendue: list) -> None:
    simple = formes.exercice_9()[nom]
    assert simple == M(attendue)
    assert est_echelonnee_simple(simple)


@pytest.mark.parametrize("nom", ["A", "B", "C", "D"])
def test_exercice_9_l_etape_intermediaire_est_echelonnee(nom: str) -> None:
    """Meme forme simple que la matrice de depart, et un journal qui y mene."""
    intermediaire = formes.echelonnees_du_9()[nom]
    assert est_echelonnee(intermediaire)
    assert not est_echelonnee(formes.MATRICES_9[nom])
    assert intermediaire.rref()[0] == formes.exercice_9()[nom]
    assert formes.journaux_du_9()[nom]


@pytest.mark.parametrize(
    ("lettre", "attendue"),
    [
        ("a", [[0, R(1, 6), -R(1, 2)], [0, R(1, 3), 0], [-1, R(5, 6), -R(1, 2)]]),
        ("b", [[-R(2, 3), -R(4, 3), 1], [-R(2, 3), R(11, 3), -2], [1, -2, 1]]),
        (
            "c",
            [
                [R(3, 7), 2, -1, -R(15, 7)],
                [R(2, 7), -1, 0, R(4, 7)],
                [R(3, 14), -R(3, 2), R(1, 2), R(10, 7)],
                [-R(1, 2), -R(1, 2), R(1, 2), 1],
            ],
        ),
        ("d", (M([[1, 1, 1, 1], [1, -1, -1, 1], [1, -1, 1, -1], [1, 1, -1, -1]]) / 4)),
    ],
)
def test_exercice_10_inverses(lettre: str, attendue: object) -> None:
    inverse = formes.exercice_10()[lettre]
    assert inverse == M(attendue)
    assert formes.MATRICES_10[lettre] * inverse == sp.eye(inverse.rows)


@pytest.mark.parametrize("ordre", [2, 3, 5])
def test_exercice_10_e_l_inverse_est_triangulaire_de_uns(ordre: int) -> None:
    inverse = formes.inverse_de_la_bidiagonale(ordre)
    assert inverse == M(ordre, ordre, lambda i, j: 1 if j >= i else 0)


@pytest.mark.parametrize("ordre", [2, 3, 4])
def test_exercice_10_f_l_inverse_de_l_antidiagonale(ordre: int) -> None:
    """`lambda_k` en ligne `k`, colonne `n + 1 - k` ; son inverse en miroir."""
    matrice, inverse = formes.antidiagonale_et_son_inverse(ordre)
    assert sp.simplify(matrice * inverse) == sp.eye(ordre)
    lambdas = sp.symbols(f"lambda1:{ordre + 1}")
    assert inverse[0, ordre - 1] == 1 / lambdas[ordre - 1]
    assert inverse[ordre - 1, 0] == 1 / lambdas[0]


def test_une_matrice_d_ordre_nul_est_refusee() -> None:
    with pytest.raises(ValueError, match="ordre non strictement positif : 0"):
        formes.inverse_de_la_bidiagonale(0)
    with pytest.raises(ValueError, match="ordre non strictement positif : -1"):
        formes.antidiagonale_et_son_inverse(-1)


def test_le_jeu_de_macros_du_chapitre_deux() -> None:
    macros = macros_du_chapitre_deux()
    assert all(nom.startswith("AlgDeux") for nom in macros)
    assert all(valeur.strip() for valeur in macros.values())
    assert r"\frac{29}{3}" in macros["AlgDeuxUn"]
    assert macros["AlgDeuxSixAntisymetrique"].count(r"\frac{1}{2}") == 2
    assert macros["AlgDeuxSixSymetrique"].count(r"\frac{5}{2}") == 2
    assert "l2 -> " not in "".join(macros.values())
    assert macros["AlgDeuxNeufCOperations"].startswith(
        r"l_{2} \to l_{2} + \tfrac{2}{3}"
    )
    assert "algebre-02-matrices" in JEUX_DE_MACROS
