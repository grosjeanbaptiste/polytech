"""`SecondOrdre` : un amortissement critique ecrit en flottant sympy reste critique.

Regression : `sp.Float(1.0) == 1` est faux, donc `regime()` classait
`xi = 1,0` en « poles reels distincts » et `indicielle` rendait `nan`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.comparaison import egales
from polytech.signaux.domain.reponses_du_second_ordre import impulsionnelle, indicielle
from polytech.signaux.domain.second_ordre import SecondOrdre
from polytech.signaux.domain.symboles import TEMPS as T

CRITIQUES = [1, 1.0, sp.Integer(1), sp.Float(1.0), sp.Rational(2, 2)]


@pytest.mark.parametrize("amortissement", CRITIQUES)
def test_un_amortissement_unite_est_critique_sous_toute_ecriture(
    amortissement: object,
) -> None:
    systeme = SecondOrdre(gain=1, constante=1, amortissement=amortissement)
    assert systeme.regime() == "poles reels confondus"


@pytest.mark.parametrize("amortissement", CRITIQUES)
def test_la_reponse_indicielle_critique_est_finie_sous_toute_ecriture(
    amortissement: object,
) -> None:
    systeme = SecondOrdre(gain=1, constante=1, amortissement=amortissement)
    attendu = 1 - (1 + T) * sp.exp(-T)
    assert indicielle(systeme).subs(T, 1).is_finite
    assert egales(indicielle(systeme), attendu)
    assert egales(impulsionnelle(systeme), T * sp.exp(-T))


def test_un_amortissement_voisin_de_un_n_est_pas_critique() -> None:
    systeme = SecondOrdre(gain=1, constante=1, amortissement=sp.Float(1.0001))
    assert systeme.regime() == "poles reels distincts"
