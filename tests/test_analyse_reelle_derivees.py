"""Chapitre 5, sections 5.1.2, 5.1.3, 5.2.2 et 5.4 : dérivées et extrema.

Source : le syllabus `Syllabus_Calcul_differentiel_et_integral_1.pdf`,
nombres dérivés à gauche et à droite (5.1.2), nombres dérivés infinis (5.1.3),
Rolle et accroissements finis (5.2.2), optimisation libre (5.4). Les formules
du syllabus sont des images que la transcription Markdown a perdues : les
fonctions d'épreuve sont les cas classiques que ces sections décrivent
(`|x|` en 0, racine cubique en 0, `x^2` sur `[0, 2]`).

Modules : `polytech.maths.domain.analyse_reelle.derivees`
(`nombres_derives`, `est_derivable_en`, `points_des_accroissements_finis`) et
`polytech.maths.domain.analyse_reelle.extrema` (`extrema_libres`, `Nature`).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.analyse_reelle.derivees import (
    est_derivable_en,
    nombres_derives,
    points_des_accroissements_finis,
)
from polytech.maths.domain.analyse_reelle.extrema import Nature, extrema_libres
from polytech.maths.domain.symboles import VARIABLE as X

RACINE_CUBIQUE = sp.sign(X) * sp.Abs(X) ** sp.Rational(1, 3)


def test_la_valeur_absolue_a_deux_nombres_derives_en_zero() -> None:
    assert nombres_derives(sp.Abs(X), 0) == (-1, 1)
    assert not est_derivable_en(sp.Abs(X), 0)


def test_un_polynome_est_derivable_et_ses_deux_nombres_coincident() -> None:
    assert nombres_derives(X**2, 1) == (2, 2)
    assert est_derivable_en(X**2, 1)


def test_la_racine_cubique_a_un_nombre_derive_infini_en_zero() -> None:
    assert nombres_derives(RACINE_CUBIQUE, 0) == (sp.oo, sp.oo)
    assert not est_derivable_en(RACINE_CUBIQUE, 0)


def test_accroissements_finis_d_un_carre() -> None:
    assert points_des_accroissements_finis(X**2, 0, 2) == (1,)


def test_accroissements_finis_avec_une_borne_non_nulle() -> None:
    assert points_des_accroissements_finis(X**2, 1, 3) == (2,)
    assert points_des_accroissements_finis(X**3, 1, 3) == (sp.sqrt(sp.Rational(13, 3)),)


def test_rolle_pour_un_sinus_entre_deux_zeros() -> None:
    assert points_des_accroissements_finis(sp.sin(X), 0, sp.pi) == (sp.pi / 2,)


def test_des_bornes_a_l_envers_levent() -> None:
    with pytest.raises(ValueError, match="a < b"):
        points_des_accroissements_finis(X**2, 2, 0)


def test_un_cubique_a_un_maximum_et_un_minimum() -> None:
    assert extrema_libres(X**3 - 3 * X) == (
        (-1, Nature.MAXIMUM),
        (1, Nature.MINIMUM),
    )


def test_le_cube_a_un_point_critique_que_la_derivee_seconde_ne_tranche_pas() -> None:
    assert extrema_libres(X**3) == ((0, Nature.INDETERMINE),)
