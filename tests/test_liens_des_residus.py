"""Le residu, enseigne en analyse complexe et employe par deux autres cours.

Mathematiques III l'enseigne (syllabus, section 9.2) ; le cours de signaux
s'en sert pour le developpement d'Heaviside (syllabus, 2.2.4), celui
d'electronique pour ses fractions simples. Il ne s'ecrit qu'une fois, dans
`polytech.maths` : ces tests font repondre les trois contextes aux memes
questions.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electronique.domain.laplace import residu_en_un_pole
from polytech.maths.domain.analyse_complexe.residus import (
    residu,
    residu_d_un_pole_simple,
)
from polytech.signaux.domain.heaviside import residu as residu_d_heaviside
from polytech.signaux.domain.heaviside import residus_d_un_pole_double
from polytech.signaux.domain.modes import residus as residus_numeriques
from polytech.signaux.domain.symboles import IMAGE as P

SIMPLES = (P + 5) / ((P + 1) * (P + 2) * (P + 4))
DOUBLE = (3 * P + 1) / ((P + 2) ** 2 * (P + 5))


@pytest.mark.parametrize("pole", [-1, -2, -4])
def test_les_trois_contextes_rendent_le_meme_residu_en_un_pole_simple(
    pole: int,
) -> None:
    """Heaviside `N / D'`, la limite du produit, et le residu de l'analyse complexe."""
    de_reference = residu(SIMPLES, P, pole)
    assert residu_d_heaviside(SIMPLES, sp.Integer(pole)) == de_reference
    assert residu_en_un_pole(SIMPLES, P, sp.Integer(pole)) == de_reference
    assert residu_d_un_pole_simple(SIMPLES, P, pole) == de_reference


def test_le_premier_coefficient_d_un_pole_double_est_le_residu() -> None:
    """`A_1`, le coefficient de `1 / (p - p_i)`, est le residu ; `A_2` ne l'est pas."""
    premier, second = residus_d_un_pole_double(DOUBLE, sp.Integer(-2))
    assert premier == residu(DOUBLE, P, -2)
    assert second == sp.limit(DOUBLE * (P + 2) ** 2, P, -2)


def test_les_residus_numeriques_sont_ceux_de_l_analyse_complexe() -> None:
    """`signaux.modes.residus`, calcul numerique, contre le calcul exact."""
    valeurs, poles = residus_numeriques([1, 5], [1, 7, 14, 8])
    for valeur, pole in zip(valeurs, poles, strict=True):
        exact = residu(SIMPLES, P, sp.nsimplify(round(pole.real)))
        assert valeur == pytest.approx(complex(exact), rel=1e-9)
