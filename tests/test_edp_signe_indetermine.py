"""Le type d'une EDP a plus de deux variables ne se devine pas.

Source : `Syllabus_Mathematiques_appliquees_2016_2017.pdf`, 9.4 (page
161) : le type suit les signes des valeurs propres de la forme
quadratique. Un signe que rien ne fixe etait compte pour negatif, et le
type sortait sans rien dire ; `type_a_deux_variables` leve dans le meme
cas.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.edp.classification import (
    Type,
    type_par_la_forme_quadratique,
)


def test_un_signe_indetermine_leve() -> None:
    k = sp.Symbol("k")
    with pytest.raises(ValueError, match="signe de la valeur propre"):
        type_par_la_forme_quadratique(sp.diag(1, k))


def test_un_parametre_dont_le_signe_est_connu_decide() -> None:
    k = sp.Symbol("k", positive=True)
    assert type_par_la_forme_quadratique(sp.diag(1, k)) is Type.ELLIPTIQUE
    assert type_par_la_forme_quadratique(sp.diag(1, -k)) is Type.HYPERBOLIQUE


def test_toutes_negatives_c_est_encore_elliptique() -> None:
    assert type_par_la_forme_quadratique(sp.diag(-1, -2, -3)) is Type.ELLIPTIQUE
