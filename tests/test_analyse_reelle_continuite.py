"""Chapitre 4, sections 4.1.5 et 4.2 : limites latérales, continuité, discontinuités.

Source : le syllabus `Syllabus_Calcul_differentiel_et_integral_1.pdf`,
définition 4.24 (continuité en un point) et définition 4.25 (discontinuités
de première et de deuxième espèce, éliminable). Les formules des exemples 4.26
sont des images que la transcription Markdown a perdues : les fonctions
d'épreuve sont celles que ces trois exemples classiques décrivent -- `sin x / x`
prolongée, un saut, une oscillation -- et non une recopie de la page.

Modules : `polytech.maths.domain.analyse_reelle.continuite`
(`limite_a_gauche`, `limite_a_droite`, `limite`, `est_continue_en`,
`classer`, `Discontinuite`) ; le prolongement par continuité (4.2.1) est
`polytech.maths.domain.series.taylor.valeur_par_continuite`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.analyse_reelle.continuite import (
    Discontinuite,
    classer,
    est_continue_en,
    limite,
    limite_a_droite,
    limite_a_gauche,
)
from polytech.maths.domain.series.taylor import valeur_par_continuite
from polytech.maths.domain.symboles import VARIABLE as X

SAUT = sp.Abs(X) / X
OSCILLATION = sp.sin(1 / X)


def test_les_limites_laterales_d_un_saut_different() -> None:
    assert limite_a_gauche(SAUT, 0) == -1
    assert limite_a_droite(SAUT, 0) == 1


def test_la_limite_existe_quand_les_deux_cotes_s_accordent() -> None:
    assert limite(sp.sin(X) / X, 0) == 1


def test_la_limite_d_un_saut_n_existe_pas() -> None:
    with pytest.raises(ValueError, match="limites laterales"):
        limite(SAUT, 0)


def test_continue_quand_la_limite_vaut_la_valeur() -> None:
    assert est_continue_en(sp.sin(X) / X, 0, valeur=1)
    assert not est_continue_en(sp.sin(X) / X, 0, valeur=2)


@pytest.mark.parametrize(
    ("expression", "valeur", "attendue"),
    [
        (sp.sin(X) / X, 1, Discontinuite.CONTINUE),
        (sp.sin(X) / X, 2, Discontinuite.ELIMINABLE),
        (SAUT, 0, Discontinuite.PREMIERE_ESPECE),
        (OSCILLATION, 0, Discontinuite.DEUXIEME_ESPECE),
        (1 / X, 0, Discontinuite.DEUXIEME_ESPECE),
    ],
)
def test_la_discontinuite_se_classe_par_les_limites_laterales(
    expression: sp.Expr, valeur: int, attendue: Discontinuite
) -> None:
    assert classer(expression, 0, valeur=valeur) is attendue


def test_le_prolongement_par_continuite_prend_la_limite_commune() -> None:
    assert valeur_par_continuite(sp.sin(X) / X, 0) == 1


def test_le_prolongement_par_continuite_refuse_un_saut() -> None:
    with pytest.raises(ValueError, match="pas derivable"):
        valeur_par_continuite(SAUT, 0)
