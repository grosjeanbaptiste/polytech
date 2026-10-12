"""Les ecritures des chapitres VII et suivants du recueil d'algebre.

Un vecteur du recueil est tantot un uplet, tantot une matrice 2 x 2 lue
ligne apres ligne, tantot un polynome donne par ses coefficients : la meme
suite de nombres se compose de trois facons.
"""

from __future__ import annotations

import re

import pytest
import sympy as sp

from polytech.maths.application import recueil_algebre_espaces as espaces
from polytech.maths.application.macros_recueil_algebre_ch07 import (
    macros_du_chapitre_sept,
)
from polytech.maths.application.registre import JEUX_DE_MACROS


def test_un_vecteur_s_ecrit_entre_parentheses() -> None:
    attendu = r"\left(1,\, -2,\, \frac{1}{2}\right)"
    assert espaces.vecteur((1, -2, sp.Rational(1, 2))) == attendu


def test_une_famille_s_ecrit_entre_accolades() -> None:
    assert espaces.famille(((1, 0), (0, 1))) == (
        r"\left\{\left(1,\, 0\right),\; \left(0,\, 1\right)\right\}"
    )
    assert espaces.famille(()) == r"\left\{\;\right\}"


def test_un_polynome_vient_de_ses_coefficients_du_plus_bas_au_plus_haut() -> None:
    assert espaces.polynome((1, -1, 0, 1)) == "X^{3} - X + 1"
    assert espaces.polynome((0, 0, 0)) == "0"


def test_une_matrice_deux_deux_se_lit_ligne_apres_ligne() -> None:
    attendue = r"\left(\begin{matrix}1 & 2\\3 & 4\end{matrix}\right)"
    assert espaces.matrice((1, 2, 3, 4)) == attendue


def test_une_matrice_qui_n_a_pas_quatre_termes_est_refusee() -> None:
    with pytest.raises(ValueError, match=" termes : une matrice 2 x 2 en a quatre"):
        espaces.matrice((1, 2, 3))


def test_une_relation_nomme_ses_vecteurs() -> None:
    assert espaces.relation((3, -2, 1), "X") == "3 X_{1} - 2 X_{2} + X_{3} = 0"


def test_une_equation_nomme_ses_inconnues() -> None:
    assert espaces.equation((0, -1, 0, 1), "xyzt") == "t - y = 0"


def test_le_jeu_de_macros_du_chapitre_sept() -> None:
    macros = macros_du_chapitre_sept()
    assert all(re.fullmatch(r"AlgSept[A-Za-z]+", nom) for nom in macros)
    assert all(valeur.strip() for valeur in macros.values())
    assert macros["AlgSeptHuit"] == "2"
    assert macros["AlgSeptTrenteTrois"] == r"\frac{11}{4}"
    assert macros["AlgSeptQuatreRelation"] == "3 X_{1} - 2 X_{2} + X_{3} = 0"
    assert "X^{3}" in macros["AlgSeptTrenteHuit"]
    assert r"\begin{matrix}" in macros["AlgSeptVingtCinq"]
    assert macros["AlgSeptSeize"] == r"2,\; 2,\; 3,\; 1"
    assert "algebre-07-espaces-vectoriels" in JEUX_DE_MACROS
