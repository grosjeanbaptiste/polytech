"""L'article « Discuter un systeme a parametre », refait par la bibliotheque.

Chaque ligne de `cursus/fiches/Mathematique.toml` pour cette notion --
les trois cas de la formule, le piege du determinant toujours nul, les
trois cas d'examen -- est recalculee par `polytech.maths` et comparee par
`polytech.maths.domain.comparaison`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.discussion import discuter
from polytech.maths.domain.algebre.discussion_par_le_rang import discuter_par_le_rang
from polytech.maths.domain.algebre.echelonnement import echelonner
from polytech.maths.domain.algebre.rang import systeme_compatible
from polytech.maths.domain.algebre.solutions import (
    Nature,
    nature_des_solutions,
    nombre_de_parametres,
)
from polytech.maths.domain.algebre.systemes import SystemeLineaire
from polytech.maths.domain.comparaison import egales

A = sp.Symbol("a", real=True)
M = sp.Symbol("m", real=True)
S2025 = sp.Matrix(
    [[A, 1 - A, 1 - A, A**2], [A, 1 + A, 1 + A, A - A**2], [1, 1, 1, 1 - A]]
)
S2022 = sp.Matrix([[1, 1, -1, 1], [3, 1, -1, 1], [1, -2, 2, M]])
S2021 = sp.Matrix(
    [
        [A, A + 1, A - 1, 2 * A + 3],
        [1, 1 + A, 1 - A, 4 * A + 1],
        [A + 1, 2 * (A + 1), 0, 6 * A + 4],
    ]
)


def _systeme(complete: sp.Matrix) -> SystemeLineaire:
    return SystemeLineaire(complete[:, :-1], complete[:, -1])


@pytest.mark.parametrize(
    ("lignes", "nature", "parametres"),
    [
        ([[1, 1, 1], [1, 1, 2]], Nature.IMPOSSIBLE, 0),
        ([[1, 1, 1], [1, -1, 0]], Nature.UNIQUE, 0),
        ([[1, 1, 1], [2, 2, 2]], Nature.INDETERMINE, 1),
    ],
)
def test_les_trois_cas_de_la_formule(
    lignes: list[list[int]], nature: Nature, parametres: int
) -> None:
    systeme = _systeme(sp.Matrix(lignes))
    assert nature_des_solutions(systeme) is nature
    assert nombre_de_parametres(systeme) == parametres
    assert systeme_compatible(
        tuple(tuple(float(x) for x in ligne[:-1]) for ligne in lignes),
        tuple(float(ligne[-1]) for ligne in lignes),
    ) is (nature is not Nature.IMPOSSIBLE)


def test_une_matrice_carree_de_determinant_non_nul_donne_une_solution_unique() -> None:
    matrice = sp.Matrix([[2, 1, 0], [1, 3, 1], [0, 1, 4]])
    assert matrice.det() != 0
    systeme = SystemeLineaire(matrice, sp.Matrix([1, 2, 3]))
    assert systeme.rang() == 3
    assert nature_des_solutions(systeme) is Nature.UNIQUE


@pytest.mark.parametrize("complete", [S2025, S2022, S2021])
def test_le_determinant_est_nul_pour_toute_valeur_et_cramer_se_tait(
    complete: sp.Matrix,
) -> None:
    systeme, parametre = _systeme(complete), (A if complete is not S2022 else M)
    assert egales(systeme.matrice.det(), 0)
    with pytest.raises(ValueError, match="determinant nul pour toute valeur"):
        discuter(systeme, parametre)


def test_les_colonnes_de_y_et_de_z_sont_proportionnelles_aux_deux_examens() -> None:
    assert egales(S2025[:, 1], S2025[:, 2])
    assert egales(S2022[:, 1], -S2022[:, 2])


def test_janvier_2021_la_troisieme_equation_est_la_somme_des_deux_premieres() -> None:
    assert egales(S2021[0, :] + S2021[1, :], S2021[2, :])


def test_janvier_2025_l_echelonnement_aboutit_a_a_fois_2a_moins_1() -> None:
    lignes_du_corrige = sp.Matrix([S2025[2, :], S2025[1, :], S2025[0, :]])
    derniere = echelonner(lignes_du_corrige, 3).matrice[2, :]
    assert egales(derniere, sp.Matrix([[0, 0, 0, A * (2 * A - 1)]]))
    discussion = discuter_par_le_rang(_systeme(S2025), A)
    assert discussion.generique is Nature.IMPOSSIBLE
    assert discussion.particulieres == {
        sp.Integer(0): Nature.INDETERMINE,
        sp.Rational(1, 2): Nature.INDETERMINE,
    }
    for valeur in discussion.particulieres:
        assert nombre_de_parametres(discussion.en(valeur)) == 1


def test_janvier_2022_compatible_pour_m_egal_moins_2_seulement_en_droite() -> None:
    derniere = echelonner(S2022, 3).matrice[2, :]
    assert egales(derniere, sp.Matrix([[0, 0, 0, M + 2]]))
    assert _systeme(S2022).rang() == 2
    discussion = discuter_par_le_rang(_systeme(S2022), M)
    assert discussion.generique is Nature.IMPOSSIBLE
    assert discussion.particulieres == {sp.Integer(-2): Nature.INDETERMINE}
    assert nombre_de_parametres(discussion.en(sp.Integer(-2))) == 1


def test_janvier_2021_systeme_4_indetermine_sauf_en_moins_un() -> None:
    discussion = discuter_par_le_rang(_systeme(S2021), A)
    assert discussion.generique is Nature.INDETERMINE
    assert discussion.particulieres[sp.Integer(-1)] is Nature.IMPOSSIBLE
