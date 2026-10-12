"""Les trois exercices que les fascicules de mathematique posent sans les resoudre.

Sources : `Mathematique/SLIDES/Slides_01_Algebre_lineaire_et_integrales.pdf`,
planches 9 et 12 (deux systemes 3 x 3), et
`Mathematique/SLIDES/Slides_02_Equations_differentielles.pdf`, planche 24
(`dy/dx + 2y = 3`, a resoudre par separation des variables).

Seule la planche 12 imprime sa solution : c'est l'oracle du second systeme.
Les deux autres reponses sont eprouvees par ce qu'elles doivent faire --
verifier le systeme, verifier l'equation -- et par un second chemin.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.application import exercices_des_fascicules as fascicules
from polytech.maths.application.registre import JEUX_DE_MACROS
from polytech.maths.domain.algebre.resolution import est_l_ensemble_des_solutions
from polytech.maths.domain.algebre.solutions import Nature
from polytech.maths.domain.algebre.systemes import SystemeLineaire
from polytech.maths.domain.differentielles.premier_ordre import INCONNUE, VARIABLE
from polytech.maths.domain.symboles import CONSTANTE

M = sp.Matrix


def test_planche_9_une_solution_unique() -> None:
    nature, solution = fascicules.systeme_pose(fascicules.PLANCHE_9)
    assert nature is Nature.UNIQUE
    assert M([[2, 3, -1], [4, 5, 2], [3, 2, -1]]) * M(list(solution)) == M([3, 8, 0])
    assert (
        tuple(M([[2, 3, -1], [4, 5, 2], [3, 2, -1]]).LUsolve(M([3, 8, 0]))) == solution
    )


def test_planche_12_la_solution_que_la_planche_imprime() -> None:
    """`{(2 - 2 alpha + beta, alpha, beta)}`."""
    nature, solution = fascicules.systeme_pose(fascicules.PLANCHE_12)
    assert nature is Nature.INDETERMINE
    alpha, beta = sp.symbols("alpha beta")
    systeme = SystemeLineaire(
        fascicules.PLANCHE_12[:, :-1], fascicules.PLANCHE_12[:, -1]
    )
    imprimee = (2 - 2 * alpha + beta, alpha, beta)
    assert est_l_ensemble_des_solutions(systeme, imprimee, (alpha, beta))
    libres = sorted(M(list(solution)).free_symbols, key=str)
    assert est_l_ensemble_des_solutions(systeme, solution, libres)


def test_planche_24_la_solution_verifie_l_equation() -> None:
    solution = fascicules.solution_de_la_planche_24()
    assert sp.simplify(sp.diff(solution, VARIABLE) + 2 * solution - 3) == 0
    assert solution.has(CONSTANTE)
    assert sp.limit(solution, VARIABLE, sp.oo) == sp.Rational(3, 2)


def test_planche_24_la_separation_mene_a_la_meme_famille() -> None:
    """Le long d'une solution, la primitive en `y` vaut `x` plus une constante."""
    primitive = fascicules.primitive_en_y()
    assert sp.simplify(sp.diff(primitive, INCONNUE) - 1 / (3 - 2 * INCONNUE)) == 0
    le_long = primitive.subs(INCONNUE, fascicules.solution_de_la_planche_24())
    assert sp.simplify(sp.diff(le_long, VARIABLE)) == 1


def test_planche_24_la_solution_constante() -> None:
    assert fascicules.solutions_constantes_de_la_planche_24() == (sp.Rational(3, 2),)
    nulle = fascicules.solution_de_la_planche_24().subs(CONSTANTE, 0)
    assert nulle == sp.Rational(3, 2)


def test_le_jeu_de_macros() -> None:
    macros = fascicules.macros_des_exercices_des_fascicules()
    assert all(nom.startswith("Fasc") and nom.isalpha() for nom in macros)
    assert all(valeur.strip() for valeur in macros.values())
    assert macros["FascNeufSolution"] == r"\left(-1,\; 2,\; 1\right)"
    assert macros["FascVingtQuatreAsymptoteNum"] == "3/2"
    assert macros["FascVingtQuatreTauxNum"] == "2"
    assert "FascDouzeSimple" in macros
    assert JEUX_DE_MACROS["exercices-des-fascicules"] is (
        fascicules.macros_des_exercices_des_fascicules
    )
