"""Les corriges des examens de janvier 2022 et 2025 : leurs valeurs.

Sources : `Mathematique/EXAMENS/Exam_2022_Janvier_Enonce.pdf` (sans corrige)
et `Exam_2025_Janvier_Corrige.pdf` (corrige officiel). Les reponses sont
recoupees par `tests/test_examens_de_mathematique.py` et
`tests/test_algebre_discussion_par_le_rang.py` ; ce fichier tient la couche
d'application des deux corriges et leurs jeux de macros.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.application import examen_2022 as vingt_deux
from polytech.maths.application import examen_2025 as vingt_cinq
from polytech.maths.application.macros_examen_2022 import macros_de_l_examen_2022
from polytech.maths.application.macros_examen_2025 import macros_de_l_examen_2025
from polytech.maths.application.registre import JEUX_DE_MACROS
from polytech.maths.domain.algebre.solutions import PARAMETRES
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T
from polytech.maths.domain.symboles import VARIABLE as X

ALPHA = PARAMETRES[0]


def test_2022_l_integrale_par_parties() -> None:
    etape, primitive, valeur = vingt_deux.integrale()
    assert etape.atoms(sp.Integral)
    assert sp.simplify(sp.diff(primitive, X) - sp.log(1 + X) / X**2) == 0
    attendue = 3 * sp.log(2) - sp.Rational(3, 2) * sp.log(3)
    assert abs(sp.N(valeur - attendue, 30)) < sp.Float("1e-25")


def test_2022_le_systeme_n_a_de_solution_qu_en_moins_deux() -> None:
    valeur, solution = vingt_deux.systeme()
    assert valeur == -2
    assert solution == (0, ALPHA + 1, ALPHA)


def test_2022_l_equation_du_second_ordre() -> None:
    image, fractions, solution = vingt_deux.equation()
    assert sp.simplify(image - sp.together(fractions)) == 0
    residu = sp.diff(solution, T, 2) - 4 * sp.diff(solution, T) + 3 * solution - T**2
    assert sp.simplify(residu) == 0
    assert solution.subs(T, 0) == 0
    assert sp.diff(solution, T).subs(T, 0) == 0


def test_2025_l_integrale_par_les_fractions_simples() -> None:
    termes, primitive, valeur = vingt_cinq.integrale()
    assert sp.simplify(sum(termes) - 4 * X**2 / (X**4 - 1)) == 0
    assert sp.simplify(sp.diff(primitive, X) - 4 * X**2 / (X**4 - 1)) == 0
    officielle = sp.log(sp.Rational(3, 2)) + 2 * sp.atan(sp.Rational(1, 7))
    assert abs(sp.N(valeur - officielle, 30)) < sp.Float("1e-25")


def test_2025_le_systeme_et_ses_deux_valeurs() -> None:
    """Le corrige officiel : `(1, y, -y)` en 0 et `(1/2, y, -y)` en 1/2."""
    solutions = vingt_cinq.systeme()
    assert solutions == {
        sp.Integer(0): (1, -ALPHA, ALPHA),
        sp.Rational(1, 2): (sp.Rational(1, 2), -ALPHA, ALPHA),
    }


def test_2025_l_equation_par_la_transformee_de_laplace() -> None:
    image, fractions, solution = vingt_cinq.equation()
    officielles = 2 / P**3 + 2 / P + 1 / (P - 7) - 3 / (P + 2)
    assert sp.simplify(fractions - officielles) == 0
    assert sp.simplify(image - sp.together(fractions)) == 0
    assert sp.simplify(solution - (T**2 + 2 + sp.exp(7 * T) - 3 * sp.exp(-2 * T))) == 0


def test_les_deux_jeux_de_macros_ne_se_melangent_pas() -> None:
    deux, cinq = macros_de_l_examen_2022(), macros_de_l_examen_2025()
    assert all(nom.startswith("ExamVingtDeux") for nom in deux)
    assert all(nom.startswith("ExamVingtCinq") for nom in cinq)
    assert all(valeur.strip() for valeur in (*deux.values(), *cinq.values()))
    assert "ExamVingtDeuxSolution" in deux
    assert "ExamVingtCinqSolution" in cinq


def test_2022_chaque_macro_parle_dans_la_variable_de_l_enonce() -> None:
    """L'enonce integre en `t` et resout en `y(x)`, a l'inverse des deux autres."""
    macros = macros_de_l_examen_2022()
    assert r"\ln{\left(t + 1 \right)}" in macros["ExamVingtDeuxPrimitive"]
    assert "x" not in macros["ExamVingtDeuxPrimitive"]
    assert "e^{3 x}" in macros["ExamVingtDeuxSolution"]
    assert "t" not in macros["ExamVingtDeuxSolution"]


def test_les_trois_examens_sont_au_registre() -> None:
    for slug in ("2021", "2022", "2025"):
        assert f"examen-{slug}-janvier" in JEUX_DE_MACROS
