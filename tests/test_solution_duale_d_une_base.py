"""Chapitre IV.4 : la solution duale associee a une base.

Oracle (`Optimisation/SLIDES/Slides_04_Definitions_proprietes_et_`
`interpretation_de_la_dualite.pdf`, pages 108-110) : `min 6 x1 + 3 x2 +
2 x3`, `x1 + x2 + x3 >= 6`, `2 x1 - 2 x3 >= 9`. Le tableau de la page 110
donne, pour chacune des neuf bases, `x(B)`, `u(B)` et leur admissibilite ;
il est recopie ici ligne par ligne.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.dual import dual_de
from polytech.optimisation.domain.ecarts_complementaires import (
    verifie_les_ecarts_complementaires,
)
from polytech.optimisation.domain.formes import FormeStandard, forme_standard
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.solution_duale import (
    BaseDuCouple,
    base_du_couple,
    bases_du_couple,
    solution_duale,
)

R = sp.Rational
SUP, INF = Sens.SUPERIEUR, Sens.INFERIEUR

PAGE_110 = [
    ((1, 2), (R(9, 2), R(3, 2), 0, 0, 0), (3, R(3, 2)), True, True),
    ((1, 3), (R(21, 4), 0, R(3, 4), 0, 0), (4, 1), True, False),
    ((1, 4), (R(9, 2), 0, 0, R(-3, 2), 0), (0, 3), False, True),
    ((1, 5), (6, 0, 0, 0, 3), (6, 0), True, False),
    ((2, 3), (0, R(21, 2), R(-9, 2), 0, 0), (3, R(1, 2)), False, True),
    ((2, 5), (0, 6, 0, 0, -9), (3, 0), False, False),
    ((3, 4), (0, 0, R(-9, 2), R(-21, 2), 0), (0, -1), False, False),
    ((3, 5), (0, 0, 6, 0, -21), (2, 0), False, True),
    ((4, 5), (0, 0, 0, -6, -9), (0, 0), False, True),
]


def _primal() -> ProgrammeGeneral:
    contraintes = (Contrainte((1, 1, 1), SUP, 6), Contrainte((2, 0, -2), SUP, 9))
    return ProgrammeGeneral(Optimisation.MIN, (6, 3, 2), contraintes)


def test_le_dual_de_l_illustration_est_celui_de_la_page_109() -> None:
    """`max 6 u1 + 9 u2`, `u1 + 2 u2 <= 6`, `u1 <= 3`, `u1 - 2 u2 <= 2`."""
    dual = dual_de(_primal())
    assert dual.optimisation is Optimisation.MAX
    assert dual.objectif == (6, 9)
    assert dual.contraintes == (
        Contrainte((1, 2), INF, 6),
        Contrainte((1, 0), INF, 3),
        Contrainte((1, -2), INF, 2),
    )


@pytest.mark.parametrize(("indices", "x", "u", "primale", "duale"), PAGE_110)
def test_chaque_ligne_du_tableau_de_la_page_110(
    indices: tuple[int, int],
    x: tuple[sp.Expr, ...],
    u: tuple[sp.Expr, ...],
    primale: bool,  # noqa: FBT001 -- une colonne du tableau du cours, pas un drapeau
    duale: bool,  # noqa: FBT001 -- une colonne du tableau du cours, pas un drapeau
) -> None:
    base = base_du_couple(forme_standard(_primal()), indices)
    assert base == BaseDuCouple(indices, x, u, primale, duale)


def test_dix_ensembles_de_deux_vecteurs_dont_neuf_bases() -> None:
    """Page 109 : `t2 = (1, 0)` et `t4 = (-1, 0)` ne forment pas une base."""
    bases = bases_du_couple(forme_standard(_primal()))
    assert [b.indices for b in bases] == [ligne[0] for ligne in PAGE_110]


def test_une_seule_base_est_optimale() -> None:
    """Page 110 : la premiere, a la fois primale et duale admissible."""
    bases = bases_du_couple(forme_standard(_primal()))
    assert [b.indices for b in bases if b.optimale] == [(1, 2)]


def test_trois_bases_primales_admissibles_et_cinq_duales() -> None:
    """Les sommets de `D1` (points 1, 2, 4) et de `D2` (points 1, 3, 5, 8, 9)."""
    bases = bases_du_couple(forme_standard(_primal()))
    assert [n for n, b in enumerate(bases, 1) if b.primale_admissible] == [1, 2, 4]
    assert [n for n, b in enumerate(bases, 1) if b.duale_admissible] == [1, 3, 5, 8, 9]


def test_la_solution_duale_d_une_base_est_cb_fois_l_inverse() -> None:
    """`u(B) = c_B B^-1` : ici `(6, 3) [[1, 1], [2, 0]]^-1 = (3, 3/2)`."""
    assert solution_duale(forme_standard(_primal()), (1, 2)) == (3, R(3, 2))


def test_pour_chaque_base_les_deux_fonctions_economiques_coincident() -> None:
    """Page 111 : `x(B)` et `u(B)` donnent la meme valeur a `z` et a `w`."""
    primal = _primal()
    for base in bases_du_couple(forme_standard(primal)):
        assert primal.valeur(base.x[:3]) == dual_de(primal).valeur(base.u)


def test_pour_chaque_base_les_ecarts_complementaires_sont_verifies() -> None:
    """Page 111 -- meme quand ni `x(B)` ni `u(B)` ne sont admissibles."""
    primal = _primal()
    for base in bases_du_couple(forme_standard(primal)):
        assert verifie_les_ecarts_complementaires(primal, base.x[:3], base.u)


def test_un_ensemble_d_indices_qui_n_est_pas_une_base_est_refuse() -> None:
    with pytest.raises(ValueError, match="n'est pas une base"):
        base_du_couple(forme_standard(_primal()), (2, 4))


def test_une_base_degeneree_reste_primale_admissible() -> None:
    """`x_B = B^-1 d >= 0` : l'egalite est permise (une variable de base nulle)."""
    forme = FormeStandard(
        couts=sp.Matrix([[-1, 0, 0, 0]]),
        matrice=sp.Matrix([[1, 1, 1, 0], [1, 0, 0, 1]]),
        seconds_membres=sp.Matrix([1, 1]),
    )
    base = base_du_couple(forme, (1, 2))
    assert base.x == (1, 0, 0, 0)
    assert base.primale_admissible
