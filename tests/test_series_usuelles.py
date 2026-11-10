"""Les six series usuelles du formulaire (§8, lignes 577-618).

`1/(1-x)` (`|x| < 1`), `e^x`, `cosh`, `sinh`, `cos`, `sin` (`x` reel). Le
chemin independant est la formule de Mac Laurin : les coefficients de la
table doivent etre ceux que la derivation de la fonction donne.
"""

import pytest
import sympy as sp

from polytech.maths.domain.series.mac_laurin import coefficients_de_mac_laurin
from polytech.maths.domain.series.series_usuelles import (
    SERIES_USUELLES,
    SerieUsuelle,
    serie,
)
from polytech.maths.domain.symboles import VARIABLE

NOMS = ("geometrique", "exponentielle", "cosh", "sinh", "cos", "sin")
RANGS = 12


def test_la_table_porte_les_six_series_du_formulaire() -> None:
    assert tuple(SERIES_USUELLES) == NOMS


@pytest.mark.parametrize("nom", NOMS)
def test_les_coefficients_sont_ceux_de_mac_laurin(nom: str) -> None:
    usuelle = serie(nom)
    attendus = coefficients_de_mac_laurin(usuelle.fonction, RANGS)
    obtenus = tuple(usuelle.coefficient(n) for n in range(RANGS))
    assert obtenus == attendus


@pytest.mark.parametrize(
    ("nom", "rayon"),
    [("geometrique", 1), *[(nom, sp.oo) for nom in NOMS[1:]]],
)
def test_le_rayon_de_convergence(nom: str, rayon: object) -> None:
    assert serie(nom).rayon() == rayon


def test_la_somme_partielle_approche_la_fonction() -> None:
    usuelle = serie("cos")
    ecart = usuelle.fonction.subs(VARIABLE, 1) - usuelle.somme_partielle(1, 10)
    """Reste de Lagrange : `|R| <= |x|^11 / 11!` pour `cos` a l'ordre 10."""
    assert abs(ecart) <= 1 / sp.factorial(11)


def test_une_serie_inconnue_est_refusee() -> None:
    with pytest.raises(KeyError, match="serie usuelle inconnue"):
        serie("tangente")


def test_la_table_ne_porte_que_des_series_usuelles() -> None:
    """`polytech.maths.domain.series.series_usuelles.SerieUsuelle` partout."""
    assert all(isinstance(s, SerieUsuelle) for s in SERIES_USUELLES.values())
