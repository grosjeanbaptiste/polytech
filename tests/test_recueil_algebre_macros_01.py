"""Les macros du corrige du chapitre I du recueil d'algebre.

Le syllabus ecrit l'unite imaginaire `i` ; le formatage des autres corriges
de mathematique l'ecrit `j`, a la maniere des electriciens. Les macros du
recueil suivent le syllabus.
"""

from __future__ import annotations

import re

import pytest
import sympy as sp

from polytech.maths.application import recueil_algebre as recueil
from polytech.maths.application.macros_recueil_algebre_ch01 import (
    macros_du_chapitre_un,
)
from polytech.maths.application.registre import JEUX_DE_MACROS


def test_l_unite_imaginaire_s_ecrit_comme_dans_le_syllabus() -> None:
    assert recueil.ecriture(3 + 4 * sp.I) == "3 + 4 i"


def test_une_liste_separe_ses_elements() -> None:
    assert recueil.liste((sp.Integer(1), sp.I)) == r"1,\; i"


def test_un_nom_de_macro_n_a_pas_de_chiffre() -> None:
    assert recueil.nom("AlgUn", 18, "c") == "AlgUnDixHuitC"
    assert recueil.nom("AlgUn", 4) == "AlgUnQuatre"


def test_un_numero_hors_de_la_table_est_refuse() -> None:
    with pytest.raises(ValueError, match="numero d'exercice hors de la table : 0"):
        recueil.nom("AlgUn", 0)
    with pytest.raises(ValueError, match="numero d'exercice hors de la table : 20"):
        recueil.nom("AlgUn", 20)


def test_l_exercice_2_publie_ses_deux_parties() -> None:
    macros = macros_du_chapitre_un()
    assert macros["AlgUnDeuxImaginaire"].startswith("4 x y")
    assert macros["AlgUnDeuxReelle"] == "x^{4} - 6 x^{2} y^{2} + y^{4}"


def test_chaque_macro_du_chapitre_un_est_nommee_et_remplie() -> None:
    macros = macros_du_chapitre_un()
    assert all(re.fullmatch(r"AlgUn[A-Za-z]+", nom) for nom in macros)
    assert all(valeur.strip() for valeur in macros.values())
    assert not any(re.search(r"(?<![a-z])j(?![a-z])", v) for v in macros.values())
    assert macros["AlgUnUnB"] == "71 - 65 i"
    assert macros["AlgUnSix"] == "-64"
    assert macros["AlgUnOnzeC"] == r"6 e^{\frac{i \pi}{30}}"


def test_le_point_de_la_figure_est_une_racine_cubique() -> None:
    points = re.findall(
        r"\((-?[\d.]+),(-?[\d.]+)\)", macros_du_chapitre_un()["AlgUnPointsQuinzeB"]
    )
    assert len(points) == 3
    for x, y in points:
        assert abs(complex(float(x), float(y)) ** 3 - (-2 + 2j)) < 1e-2


def test_le_chapitre_est_au_registre() -> None:
    assert "algebre-01-nombres-complexes" in JEUX_DE_MACROS
