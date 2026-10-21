"""Syllabus d'algebre IV.1 et IV.2 : deux gestes que le code ne portait pas.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf` (pages imprimees).

- IV.1, pages 48 et 49 : « on peut aussi proceder par deductions
  successives ». Diviser `3x^5 - 6x^4 + x^3 + 5x^2 - 3x - 4` par `x - 1` :
  le cours deduit `a = -3`, `b = -2`, `c = 3`, `d = 0`, et le reste `-4` ;
- IV.2 4), page 51 : « regrouper des termes afin de mettre en evidence un
  facteur commun dans chaque groupe ». `3x^3 + 7x^2 - 7x - 3 =
  (3x^3 - 3) + (7x^2 - 7x) = (x - 1)(3x^2 + 10x + 3) = (x - 1)(x + 3)(3x + 1)`.

Les notes de 2011 (`Syllabus_Notes_de_cours_2011.pdf`, page 45) divisent
`3x^3 + 2x + 1` par `x^2 - x - 1` : quotient `3x + 3`, reste `8x + 4`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.polynomes_exacts import (
    division,
    par_deductions_successives,
)
from polytech.maths.domain.algebre.regroupement import par_regroupement
from polytech.maths.domain.symboles import VARIABLE as X

DIVIDENDE = 3 * X**5 - 6 * X**4 + X**3 + 5 * X**2 - 3 * X - 4


def test_les_deductions_rendent_les_coefficients_du_cours() -> None:
    coefficients, _ = par_deductions_successives(DIVIDENDE, X - 1)
    assert coefficients == (3, -3, -2, 3, 0)


def test_les_deductions_rendent_le_reste_du_cours() -> None:
    _, reste = par_deductions_successives(DIVIDENDE, X - 1)
    assert reste == -4


def test_les_deductions_concordent_avec_la_division_ecrite() -> None:
    """Second chemin : la division euclidienne de sympy."""
    dividende = 3 * X**3 + 2 * X + 1
    coefficients, reste = par_deductions_successives(dividende, X**2 - X - 1)
    quotient = sum(
        c * X ** (len(coefficients) - 1 - i) for i, c in enumerate(coefficients)
    )
    assert (sp.expand(quotient), reste) == division(dividende, X**2 - X - 1)


def test_la_division_des_notes_de_2011() -> None:
    assert division(3 * X**3 + 2 * X + 1, X**2 - X - 1) == (3 * X + 3, 8 * X + 4)


def test_un_diviseur_non_unitaire_divise_ses_coefficients() -> None:
    coefficients, reste = par_deductions_successives(6 * X**2 + 5 * X + 1, 2 * X + 1)
    assert (coefficients, reste) == ((3, 1), 0)


def test_les_deductions_refusent_un_diviseur_constant() -> None:
    with pytest.raises(ValueError, match="degre au moins 1"):
        par_deductions_successives(DIVIDENDE, sp.Integer(2))


def test_un_dividende_de_degre_moindre_n_a_que_son_reste() -> None:
    assert par_deductions_successives(X + 1, X**2 + 1) == ((), X + 1)


def test_le_regroupement_du_cours_met_x_moins_un_en_evidence() -> None:
    commun, cofacteur = par_regroupement((3 * X**3 - 3, 7 * X**2 - 7 * X))
    assert sp.expand(commun) == X - 1
    assert sp.expand(cofacteur) == 3 * X**2 + 10 * X + 3


def test_le_regroupement_mene_a_la_factorisation_du_cours() -> None:
    commun, cofacteur = par_regroupement((3 * X**3 - 3, 7 * X**2 - 7 * X))
    assert sp.factor(commun * cofacteur) == sp.factor((X - 1) * (X + 3) * (3 * X + 1))
    assert sp.expand(commun * cofacteur) == 3 * X**3 + 7 * X**2 - 7 * X - 3


def test_un_regroupement_sans_facteur_commun_leve() -> None:
    with pytest.raises(ValueError, match="aucun facteur commun"):
        par_regroupement((X**2 + 1, X + 2))


def test_un_seul_groupe_ne_regroupe_rien() -> None:
    with pytest.raises(ValueError, match="deux groupes"):
        par_regroupement((X**2 - 1,))
