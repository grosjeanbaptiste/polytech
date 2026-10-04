"""Mathematique, planches 3 a 19 : les systemes d'equations algebriques lineaires.

Oracles (`Mathematique/SLIDES/Slides_01_Algebre_lineaire_et_integrales.pdf`) :
le systeme `2x + y = 1`, `3x + 7y = -2` resolu de quatre manieres
(planches 4 a 7), solution `(9/11, -7/11)` ; les systemes impossibles et
indetermines des planches 8, 10, 12, 17 et 18 ; la methode de Cramer
generalisee de la planche 16 ; la condition de compatibilite de la
planche 19.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.methodes import (
    inverse_d_ordre_deux,
    par_cramer,
    par_l_inverse,
    par_le_pivot,
)
from polytech.maths.domain.algebre.solutions import (
    PARAMETRES,
    Nature,
    cramer_generalise,
    nature_des_solutions,
    nombre_de_parametres,
    solution_generale,
)
from polytech.maths.domain.algebre.substitution import par_substitution
from polytech.maths.domain.algebre.systemes import SystemeLineaire

R = sp.Rational
ALPHA, BETA = PARAMETRES[:2]
SOLUTION = sp.Matrix([R(9, 11), R(-7, 11)])
SOURCE = "Mathematique/SLIDES/Slides_01_Algebre_lineaire_et_integrales.pdf"


def _systeme(lignes: list[list[int]], second: list[int]) -> SystemeLineaire:
    return SystemeLineaire(sp.Matrix(lignes), sp.Matrix(second))


def _egaux(obtenu: tuple[sp.Expr, ...], attendu: tuple[sp.Expr, ...]) -> bool:
    """L'egalite composante par composante, a une reecriture algebrique pres."""
    paires = zip(obtenu, attendu, strict=True)
    return all(sp.simplify(a - b) == 0 for a, b in paires)


def _deux() -> SystemeLineaire:
    """Planche 3 : `2x + y = 1`, `3x + 7y = -2`."""
    return _systeme([[2, 1], [3, 7]], [1, -2])


def _exercice() -> SystemeLineaire:
    """Planche 9 : l'exercice 3 x 3, dont la planche ne donne pas la solution."""
    return _systeme([[2, 3, -1], [4, 5, 2], [3, 2, -1]], [3, 8, 0])


def test_l_inverse_d_une_matrice_d_ordre_deux() -> None:
    """Planche 6 : `1/(ad - bc) [[d, -b], [-c, a]]`, soit `1/11 [[7, -1], [-3, 2]]`."""
    inverse = inverse_d_ordre_deux(sp.Matrix([[2, 1], [3, 7]]))
    assert inverse == sp.Matrix([[7, -1], [-3, 2]]) / 11
    assert inverse * sp.Matrix([[2, 1], [3, 7]]) == sp.eye(2)


def test_une_matrice_de_determinant_nul_n_a_pas_d_inverse() -> None:
    """Planche 6 : « il faut bien sur que le determinant soit non nul »."""
    with pytest.raises(ValueError, match="determinant nul"):
        inverse_d_ordre_deux(sp.Matrix([[3, -2], [-6, 4]]))


def test_l_inverse_d_ordre_deux_refuse_une_autre_taille() -> None:
    with pytest.raises(ValueError, match="matrice 2 x 2 attendue"):
        inverse_d_ordre_deux(sp.eye(3))


def test_les_quatre_methodes_donnent_la_meme_solution() -> None:
    """Planches 4 a 7 : substitution, Gauss, matrices, Cramer."""
    assert par_l_inverse(_deux()) == SOLUTION
    assert par_cramer(_deux()) == SOLUTION
    assert par_le_pivot(_deux()).solution == SOLUTION
    assert par_substitution(_deux()).conclusion == "x = 9/11, y = -7/11"


def test_la_substitution_suit_les_etapes_de_la_planche_4() -> None:
    raisonnement = par_substitution(_deux())
    assert [e.affirmation for e in raisonnement.etapes] == [
        "y = 1 - 2*x",
        "11*x = 9",
        "x = 9/11",
        "y = -7/11",
        "la solution verifie les deux equations",
    ]
    assert raisonnement.sources == (f"{SOURCE}, planche 4",)
    assert "3*x + 7*(1 - 2*x) = -2" in raisonnement.etapes[1].justification


def test_la_substitution_isole_l_inconnue_dont_le_coefficient_est_simple() -> None:
    """`x + 2y = 5`, `3x - y = 1` : on isole `x` de la premiere equation."""
    raisonnement = par_substitution(_systeme([[1, 2], [3, -1]], [5, 1]))
    assert raisonnement.etapes[0].affirmation == "x = 5 - 2*y"
    assert raisonnement.conclusion == "x = 1, y = 2"


def test_la_substitution_ne_vaut_que_pour_deux_equations_a_deux_inconnues() -> None:
    with pytest.raises(ValueError, match="systeme 2 x 2 attendu"):
        par_substitution(_exercice())


def test_le_pivot_de_gauss_garde_l1_et_remplace_l2() -> None:
    """Planche 5 : `L2 <- 2 L2 - 3 L1`, d'ou le systeme triangulaire `11 y = -7`."""
    echelonnement = par_le_pivot(_deux())
    assert echelonnement.operations == ("L_2 <- 2 L_2 - 3 L_1",)
    assert echelonnement.triangulaire.matrice == sp.Matrix([[2, 1], [0, 11]])
    assert echelonnement.triangulaire.second_membre == sp.Matrix([1, -7])


def test_les_formules_de_cramer_remplacent_une_colonne_par_le_second_membre() -> None:
    """Planche 7 : `x = |1 1; -2 7| / 11 = 9/11`, `y = |2 1; 3 -2| / 11 = -7/11`."""
    assert sp.Matrix([[1, 1], [-2, 7]]).det() == 9
    assert sp.Matrix([[2, 1], [3, -2]]).det() == -7
    assert par_cramer(_deux()) == sp.Matrix([R(9, 11), R(-7, 11)])


def test_cramer_refuse_un_determinant_nul_ou_un_systeme_non_carre() -> None:
    """Planche 14 : « peut-on appliquer Cramer ? (le systeme n'est pas carre !) »."""
    with pytest.raises(ValueError, match="determinant nul"):
        par_cramer(_systeme([[3, -2], [-6, 4]], [1, 3]))
    with pytest.raises(ValueError, match="systeme carre attendu"):
        par_cramer(_systeme([[3, -2, 7], [-6, 4, 3]], [1, 2]))


def test_l_exercice_de_la_planche_9_par_trois_chemins() -> None:
    """Aucune solution n'est donnee : les trois methodes doivent s'accorder."""
    systeme = _exercice()
    assert systeme.matrice.det() == 19
    par_inverse, par_determinants = par_l_inverse(systeme), par_cramer(systeme)
    assert par_inverse == par_determinants == par_le_pivot(systeme).solution
    assert systeme.matrice * par_inverse == systeme.second_membre


def test_le_pivot_echelonne_un_systeme_trois_par_trois() -> None:
    echelonnement = par_le_pivot(_exercice())
    triangulaire = echelonnement.triangulaire.matrice
    assert all(triangulaire[i, j] == 0 for i in range(3) for j in range(i))
    assert echelonnement.operations[0] == "L_2 <- 2 L_2 - 4 L_1"
    assert len(echelonnement.operations) == 3


def test_le_pivot_echange_deux_lignes_quand_le_pivot_est_nul() -> None:
    systeme = _systeme([[0, 1], [1, 1]], [2, 3])
    echelonnement = par_le_pivot(systeme)
    assert echelonnement.operations == ("L_1 <-> L_2",)
    assert echelonnement.solution == sp.Matrix([1, 2])


def test_le_pivot_refuse_un_systeme_qui_n_a_pas_une_solution_unique() -> None:
    with pytest.raises(ValueError, match="pas de solution unique"):
        par_le_pivot(_systeme([[3, -2], [-6, 4]], [1, 3]))


@pytest.mark.parametrize(
    ("lignes", "second", "nature", "parametres"),
    [
        ([[2, 1], [3, 7]], [1, -2], Nature.UNIQUE, 0),
        ([[3, -2], [-6, 4]], [1, 3], Nature.IMPOSSIBLE, 0),
        ([[3, -2], [-6, 4]], [1, -2], Nature.INDETERMINE, 1),
        ([[0, 0], [0, 0]], [0, 0], Nature.INDETERMINE, 2),
        ([[3, -1, 1], [5, 1, -2], [-2, -2, 3]], [2, 6, 7], Nature.IMPOSSIBLE, 0),
        ([[3, -1, 1], [5, 1, -2], [-2, -2, 3]], [2, 6, -4], Nature.INDETERMINE, 1),
        ([[1, 2, -1], [-1, -2, 1], [2, 4, -2]], [2, -2, 4], Nature.INDETERMINE, 2),
        ([[3, -2, 7], [-6, 4, 3]], [1, 2], Nature.INDETERMINE, 1),
    ],
)
def test_la_nature_des_solutions_des_systemes_du_cours(
    lignes: list[list[int]],
    second: list[int],
    nature: Nature,
    parametres: int,
) -> None:
    """Planches 3, 8, 10, 12, 13, 17, 18 : vide, un point, une droite, un plan."""
    systeme = _systeme(lignes, second)
    assert nature_des_solutions(systeme) is nature
    assert nombre_de_parametres(systeme) == parametres


def test_la_solution_generale_de_la_planche_12() -> None:
    """`{(2 - 2 alpha + beta, alpha, beta)}` : deux parametres."""
    systeme = _systeme([[1, 2, -1], [-1, -2, 1], [2, 4, -2]], [2, -2, 4])
    assert _egaux(solution_generale(systeme), (2 - 2 * ALPHA + BETA, ALPHA, BETA))


def test_la_solution_generale_d_un_systeme_regulier_n_a_pas_de_parametre() -> None:
    assert solution_generale(_deux()) == (R(9, 11), R(-7, 11))


def test_un_systeme_impossible_n_a_pas_de_solution_generale() -> None:
    with pytest.raises(ValueError, match="systeme impossible"):
        solution_generale(_systeme([[3, -2], [-6, 4]], [1, 3]))


def test_cramer_generalise_choisit_une_sous_matrice_reguliere() -> None:
    """Planches 13 a 16 : `x, y` ne conviennent pas (determinant nul), `x, z` oui."""
    systeme = _systeme([[3, -2, 7], [-6, 4, 3]], [1, 2])
    reduction = cramer_generalise(systeme)
    assert (reduction.lignes, reduction.colonnes) == ((0, 1), (0, 2))
    assert reduction.sous_matrice.det() == 51
    assert _egaux(reduction.solution, ((34 * ALPHA - 11) / 51, ALPHA, R(4, 17)))


def test_cramer_generalise_neglige_l_equation_qui_n_intervient_pas() -> None:
    """Planche 18 : le systeme simplement indetermine, `z = alpha`."""
    systeme = _systeme([[3, -1, 1], [5, 1, -2], [-2, -2, 3]], [2, 6, -4])
    reduction = cramer_generalise(systeme)
    assert (reduction.lignes, reduction.colonnes) == ((0, 1), (0, 1))
    assert _egaux(reduction.solution, (1 + ALPHA / 8, 1 + 11 * ALPHA / 8, ALPHA))


def test_cramer_generalise_verifie_les_equations_negligees() -> None:
    """Planche 16, etape 5 : sans elle, la planche 17 aurait une « solution »."""
    systeme = _systeme([[3, -1, 1], [5, 1, -2], [-2, -2, 3]], [2, 6, 7])
    with pytest.raises(ValueError, match="ne satisfont pas le systeme"):
        cramer_generalise(systeme)


def test_cramer_generalise_et_la_solution_generale_decrivent_le_meme_ensemble() -> None:
    systeme = _systeme([[3, -1, 1], [5, 1, -2], [-2, -2, 3]], [2, 6, -4])
    point = sp.Matrix(cramer_generalise(systeme).solution)
    assert sp.simplify(systeme.matrice * point - systeme.second_membre) == sp.zeros(
        3, 1
    )
    autre = sp.Matrix(solution_generale(systeme))
    assert sp.simplify(systeme.matrice * autre - systeme.second_membre) == sp.zeros(
        3, 1
    )


def test_un_systeme_homogene_est_toujours_compatible() -> None:
    """Planche 19 : `x = 0` est toujours solution."""
    homogene = _systeme([[3, -2], [-6, 4]], [0, 0])
    assert nature_des_solutions(homogene) is Nature.INDETERMINE
    assert homogene.est_homogene()


def test_la_substitution_d_un_systeme_sans_solution_unique_le_dit() -> None:
    """Planche 10 : `3 x1 - 2 x2 = 1`, `-6 x1 + 4 x2 = 3` -- l'inconnue disparait."""
    with pytest.raises(ValueError, match="determinant nul"):
        par_substitution(_systeme([[3, -2], [-6, 4]], [1, 3]))
