"""Le chapitre V du recueil d'algebre, « Les determinants », refait.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, exercices 1 a 8 de
la section V.6 ; l'oracle est « Solutions du 5eme chapitre » (page 63 du
syllabus). Les exercices 3, 4 et 6 sont des demonstrations : le syllabus
n'en imprime rien, et ils sont verifies ici sur des cas.
"""

from __future__ import annotations

import re

import pytest
import sympy as sp

from polytech.maths.application import recueil_algebre_ch05 as chapitre
from polytech.maths.application.macros_recueil_algebre_ch05 import (
    macros_du_chapitre_cinq,
)
from polytech.maths.application.registre import JEUX_DE_MACROS

A, B, C, D = chapitre.LETTRES
X, LAM = chapitre.INCONNUE, chapitre.LAMBDA


def test_exercice_1_les_determinants_numeriques() -> None:
    reponses = chapitre.exercice_1()
    assert reponses["a"] == -8
    assert reponses["c"] == 72
    assert reponses["d"] == -14


def test_exercice_1_les_determinants_litteraux() -> None:
    reponses = chapitre.exercice_1()
    attendu_b = B * C * D + A * C * D + A * B * D + A * B * C + A * B * C * D
    assert sp.expand(reponses["b"] - attendu_b) == 0
    assert sp.expand(reponses["e"] - (A - B) * (B - C) * (C - A)) == 0


@pytest.mark.parametrize("n", [2, 3, 4, 7])
def test_exercice_1f_la_tridiagonale_vaut_n_plus_un(n: int) -> None:
    assert chapitre.tridiagonale(n) == n + 1


def test_exercice_2_par_le_produit_avec_la_transposee() -> None:
    produit, determinant = chapitre.exercice_2()
    assert produit == (A**2 + 3) * sp.eye(4)
    assert sp.expand(determinant - (A**2 + 3) ** 2) == 0


@pytest.mark.parametrize("n", [2, 3, 4])
def test_exercice_3_vandermonde(n: int) -> None:
    determinant, produit = chapitre.vandermonde(n)
    assert sp.expand(determinant - produit) == 0


def test_exercice_4_un_determinant_double() -> None:
    gauche, droite = chapitre.exercice_4()
    assert sp.expand(gauche - 2 * droite) == 0
    somme, simple = chapitre.matrices_du_4()
    assert somme[:, 0] == simple[:, 1] + simple[:, 2]
    assert somme[:, 2] == simple[:, 0] + simple[:, 1]


@pytest.mark.parametrize(("n", "attendu"), [(1, 1), (2, -2), (3, 0), (4, 0), (6, 0)])
def test_exercice_5_les_entiers_ranges_en_carre(n: int, attendu: int) -> None:
    assert chapitre.exercice_5(n) == attendu


MATRICES_DU_6 = [
    sp.Matrix(2, 2, sp.symbols("m1:5")),
    sp.Matrix(3, 3, sp.symbols("m1:10")),
    sp.Matrix([[2, 1, 0, 3], [1, -1, 4, 0], [0, 2, 1, 1], [5, 0, -2, 1]]),
]
"""Deux matrices generales, et une d'ordre 4 chiffree : le general y est lent."""


@pytest.mark.parametrize("matrice", MATRICES_DU_6)
def test_exercice_6_le_determinant_de_la_matrice_des_cofacteurs(
    matrice: sp.Matrix,
) -> None:
    cofacteurs, determinant = chapitre.exercice_6(matrice)
    assert determinant != 0
    assert sp.expand(cofacteurs - determinant ** (matrice.rows - 1)) == 0


def test_exercice_7_les_valeurs_qui_annulent() -> None:
    determinant, racines = chapitre.exercice_7()
    assert sp.expand(determinant - (LAM - 4) * (LAM + 2) ** 2) == 0
    assert racines == {sp.Integer(-2): 2, sp.Integer(4): 1}


def test_exercice_8_les_valeurs_qui_annulent() -> None:
    determinant, racines = chapitre.exercice_8()
    assert sp.expand(determinant + 4 * X**2 * (X - 2) * (X + 2)) == 0
    assert set(racines) == {sp.Integer(0), sp.Integer(2), sp.Integer(-2)}


def test_un_ordre_non_strictement_positif_est_refuse() -> None:
    with pytest.raises(ValueError, match="ordre non strictement positif : 0"):
        chapitre.tridiagonale(0)
    with pytest.raises(ValueError, match="ordre non strictement positif : -2"):
        chapitre.exercice_5(-2)


def test_le_jeu_de_macros_du_chapitre_cinq() -> None:
    macros = macros_du_chapitre_cinq()
    assert all(re.fullmatch(r"AlgCinq[A-Za-z]+", nom) for nom in macros)
    assert all(valeur.strip() for valeur in macros.values())
    assert macros["AlgCinqUnA"] == "-8"
    assert macros["AlgCinqHuitRacines"] == r"-2,\; 0,\; 2"
    assert macros["AlgCinqSeptRacines"] == r"-2,\; 4"
    assert "a_{3}" in macros["AlgCinqTroisTrois"]
    assert macros["AlgCinqUnFSuite"] == r"3,\; 4,\; 5,\; 6"
    assert macros["AlgCinqCinqSuite"] == r"1,\; -2,\; 0,\; 0"
    points = re.findall(r"\((-?[\d.]+),(-?[\d.]+)\)", macros["AlgCinqCourbeSept"])
    assert ("-2", "0") in points
    assert ("4", "0") in points
    assert "algebre-05-determinants" in JEUX_DE_MACROS
