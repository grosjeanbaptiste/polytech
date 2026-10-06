"""Mathematiques II, section 2 : les exemples des planches, par la bibliotheque.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que les criteres de `polytech.maths.domain.series.numeriques`
portent ceux de la section 2, en recopiant ses exemples.

Source : `CalculDifferentielIntegral2/SLIDES/Slides_02_Series_numeriques.pdf`
(53 pages pour 51 planches : les pages citees sont celles du PDF).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.series.entieres import INDICE, SerieEntiere
from polytech.maths.domain.series.numeriques import (
    Verdict,
    converge,
    converge_absolument,
    critere_de_la_racine,
    critere_du_quotient,
    est_alternee_convergente,
    somme_de_la_serie_geometrique,
    somme_partielle,
    sont_equivalentes,
)
from polytech.maths.domain.series.produit import produit_de_cauchy

K = sp.Symbol("k", integer=True, positive=True)
HARMONIQUE = 1 / K


@pytest.mark.parametrize(
    ("rang", "arrondie"),
    [(1, 1.00), (5, 2.28), (10, 2.93), (50, 4.50), (100, 5.19), (500, 6.79)],
)
def test_les_sommes_partielles_de_la_serie_harmonique(
    rang: int, arrondie: float
) -> None:
    """Page 16 : le tableau des `s_n`, a deux decimales."""
    assert round(float(somme_partielle(HARMONIQUE, K, rang)), 2) == arrondie


def test_la_serie_geometrique_de_raison_un_demi_vaut_deux() -> None:
    """Pages 10 et 11 : `1 / (1 - rho)`, et le paradoxe de Zenon."""
    assert somme_de_la_serie_geometrique(sp.Rational(1, 2)) == 2


@pytest.mark.parametrize(
    ("terme", "nature"),
    [(HARMONIQUE, False), ((-1) ** K, False), ((-1) ** K / K, True)],
    ids=["harmonique", "alternee", "harmonique alternee"],
)
def test_la_nature_des_trois_series_des_pages_12_a_15(
    terme: sp.Expr, *, nature: bool
) -> None:
    """Pages 13 a 15 : « diverge », « diverge », « converge »."""
    assert converge(terme, K) is nature


def test_la_serie_harmonique_alternee_converge_sans_converger_absolument() -> None:
    """Pages 18 et 22 : la convergence simple n'entraine pas l'absolue."""
    assert est_alternee_convergente(HARMONIQUE, K)
    assert not converge_absolument((-1) ** K / K, K)


def test_les_deux_exemples_d_equivalence_de_la_page_33() -> None:
    """Page 33 : `1 / racine3(7 k2 + pi)` diverge comme `1 / k^(2/3)`.

    Le second exemple a une limite nulle : `k2 e^(-k)` n'est pas equivalente
    a `1 / k2`, elle est dominee par elle, et converge.
    """
    reference = 1 / K ** sp.Rational(2, 3)
    assert sont_equivalentes(1 / sp.cbrt(7 * K**2 + sp.pi), reference, K)
    assert not converge(reference, K)
    dominee = K**2 * sp.exp(-K)
    assert not sont_equivalentes(dominee, 1 / K**2, K)
    assert converge(dominee, K)


@pytest.mark.parametrize(
    ("terme", "verdict"),
    [
        (K**2 * sp.exp(-K), Verdict.CONVERGE),
        (HARMONIQUE, Verdict.INDECIS),
        (2**K / K, Verdict.DIVERGE),
    ],
)
def test_les_trois_issues_du_quotient_et_de_la_racine(
    terme: sp.Expr, verdict: Verdict
) -> None:
    """Pages 34 et 37 : converge, « pas de conclusion », diverge, selon `alpha`."""
    assert critere_du_quotient(terme, K)[1] is verdict
    assert critere_de_la_racine(terme, K)[1] is verdict


def test_le_produit_au_sens_de_cauchy_de_la_page_47() -> None:
    """Page 47 : `z_m = somme de k = 0 a m de x_k y_(m-k)`."""
    x, y = SerieEntiere(INDICE + 1), SerieEntiere(sp.Integer(2) ** INDICE)
    z = produit_de_cauchy(x, y, 2)
    assert z[0] == x.coefficient(0) * y.coefficient(0)
    assert z[1] == x.coefficient(0) * y.coefficient(1) + x.coefficient(
        1
    ) * y.coefficient(0)
    assert z[2] == sum(x.coefficient(k) * y.coefficient(2 - k) for k in range(3))
