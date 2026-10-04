"""Chapitre I.4.3 d'optimisation : bases, solutions de base, sommets.

Oracles : les deux tableaux du cours
(`Optimisation/SLIDES/Slides_01_Les_bases_de_la_programmation_lineaire.pdf`,
pages 43 a 45), recopies ligne par ligne -- les dix bases de l'exemple
`max 6 x1 + 5 x2`, et les deux sous-matrices non regulieres du second.
Les indices y sont ceux du cours, a partir de 1.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.bases import (
    SolutionDeBase,
    solutions_de_base,
    sommets,
)
from polytech.optimisation.domain.formes import FormeStandard

R = sp.Rational


def _exemple_1() -> FormeStandard:
    """x1 + x2 + x3 = 8, -2 x1 + 3 x2 + x4 = 6, x1 - x2 + x5 = 2 (page 43)."""
    return FormeStandard(
        couts=sp.Matrix([[-6, -5, 0, 0, 0]]),
        matrice=sp.Matrix([[1, 1, 1, 0, 0], [-2, 3, 0, 1, 0], [1, -1, 0, 0, 1]]),
        seconds_membres=sp.Matrix([8, 6, 2]),
    )


TABLEAU_DU_COURS: dict[tuple[int, ...], tuple[tuple[object, ...], bool]] = {
    (1, 2, 3): ((12, 10, -14), False),
    (1, 2, 4): ((5, 3, 7), True),
    (1, 2, 5): ((R(18, 5), R(22, 5), R(14, 5)), True),
    (1, 3, 4): ((2, 6, 10), True),
    (1, 3, 5): ((-3, 11, 5), False),
    (1, 4, 5): ((8, 22, -6), False),
    (2, 3, 4): ((-2, 10, 12), False),
    (2, 3, 5): ((2, 6, 4), True),
    (2, 4, 5): ((8, -18, 10), False),
    (3, 4, 5): ((8, 6, 2), True),
}


def _par_indices(forme: FormeStandard) -> dict[tuple[int, ...], SolutionDeBase]:
    return {tuple(i + 1 for i in s.indices): s for s in solutions_de_base(forme)}


def test_les_dix_sous_matrices_de_l_exemple_sont_des_bases() -> None:
    assert set(_par_indices(_exemple_1())) == set(TABLEAU_DU_COURS)


@pytest.mark.parametrize("indices", sorted(TABLEAU_DU_COURS))
def test_chaque_solution_de_base_est_celle_du_tableau(indices: tuple[int, ...]) -> None:
    valeurs, admissible = TABLEAU_DU_COURS[indices]
    solution = _par_indices(_exemple_1())[indices]
    assert solution.valeurs_de_base == tuple(sp.nsimplify(v) for v in valeurs)
    assert solution.admissible is admissible


def test_les_variables_hors_base_sont_nulles() -> None:
    solution = _par_indices(_exemple_1())[(1, 2, 4)]
    assert list(solution.point) == [5, 3, 0, 7, 0]


def _exemple_2() -> FormeStandard:
    """x1 <= 4, x2 <= 3, x1 + 2 x2 <= 8, mis sous forme standard (page 45)."""
    return FormeStandard(
        couts=sp.Matrix([[-2, -5, 0, 0, 0]]),
        matrice=sp.Matrix([[1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 2, 0, 0, 1]]),
        seconds_membres=sp.Matrix([4, 3, 8]),
    )


def test_deux_sous_matrices_singulieres_ne_sont_pas_des_bases() -> None:
    """{1,3,5} et {2,4,5} : des droites paralleles, pas d'intersection."""
    bases = set(_par_indices(_exemple_2()))
    assert (1, 3, 5) not in bases
    assert (2, 4, 5) not in bases
    assert len(bases) == 8


def test_les_bases_admissibles_du_second_exemple() -> None:
    admissibles = {i for i, s in _par_indices(_exemple_2()).items() if s.admissible}
    assert admissibles == {(3, 4, 5), (2, 3, 5), (1, 2, 3), (1, 2, 4), (1, 4, 5)}


def test_les_sommets_sont_les_solutions_de_base_admissibles() -> None:
    """Propriete fondamentale : un sommet par solution de base admissible."""
    attendus = {(0, 0), (2, 0), (5, 3), (R(18, 5), R(22, 5)), (0, 2)}
    assert {tuple(p) for p in sommets(_exemple_1(), variables=2)} == attendus


def _degenere() -> FormeStandard:
    """x1 <= 1, x2 <= 1, x1 + x2 <= 2 : trois droites passent par (1, 1)."""
    return FormeStandard(
        couts=sp.Matrix([[-1, -1, 0, 0, 0]]),
        matrice=sp.Matrix([[1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 1, 0, 0, 1]]),
        seconds_membres=sp.Matrix([1, 1, 2]),
    )


def test_une_base_avec_une_variable_de_base_nulle_est_degeneree() -> None:
    solution = _par_indices(_degenere())[(1, 2, 3)]
    assert solution.degeneree


def test_un_meme_sommet_peut_correspondre_a_plusieurs_bases() -> None:
    """Remarque 4 du cours : la degenerescence (renvoi au II.6)."""
    au_sommet = [
        s for s in solutions_de_base(_degenere()) if list(s.point[:2]) == [1, 1]
    ]
    assert len(au_sommet) == 3


def test_un_systeme_redondant_est_refuse() -> None:
    """Le cours suppose `r(T) = m` ; une equation combinaison des autres est refusee."""
    redondant = FormeStandard(
        couts=sp.Matrix([[1, 1]]),
        matrice=sp.Matrix([[1, 1], [2, 2]]),
        seconds_membres=sp.Matrix([1, 2]),
    )
    with pytest.raises(ValueError, match="systeme redondant"):
        solutions_de_base(redondant)


def test_une_solution_degeneree_reste_admissible() -> None:
    """Une variable de base nulle n'est pas negative : le sommet (1, 1) existe."""
    assert _par_indices(_degenere())[(1, 2, 3)].admissible


def test_un_sommet_degenere_n_est_rendu_qu_une_fois() -> None:
    """Le carre a quatre sommets, bien que (1, 1) vienne de trois bases."""
    assert len(sommets(_degenere(), variables=2)) == 4
