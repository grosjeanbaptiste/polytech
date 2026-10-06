"""Mathematiques II, section 3 : les exemples des planches, par la bibliotheque.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que `polytech.maths.domain.series` porte la convergence simple, la
convergence uniforme et le critere de Weierstrass tels que la section 3 les
enonce, en recopiant ses exemples.

Sources : `CalculDifferentielIntegral2/SLIDES/
Slides_03_Suites_et_series_de_fonctions.pdf` (35 pages) et son edition
`..._2025_2026.pdf` (45 pages), citee « 2025 ».
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.series.suites_de_fonctions import (
    distances_uniformes,
    limite_simple,
)
from polytech.maths.domain.series.uniforme import (
    est_majoree_par,
    serie_majorante_convergente,
)
from polytech.maths.domain.symboles import VARIABLE

N = sp.Symbol("n", integer=True, positive=True)
X = VARIABLE
PUISSANCE = X**N


@pytest.mark.parametrize(
    ("terme", "point", "limite"),
    [
        ((2 * N * X**2 - 1) / (N * X + X**2), 3, 6),
        (N * sp.sin(X / N), sp.Rational(5, 2), sp.Rational(5, 2)),
        (N / (1 + N * (1 + X)), 3, sp.Rational(1, 4)),
        (sp.sin(N * X) / sp.sqrt(N), 1, 0),
        (N**2 * X * (1 - X**2) ** N, sp.Rational(1, 2), 0),
    ],
    ids=["page 5", "2025 page 8", "2025 page 24", "page 9", "page 10"],
)
def test_la_limite_simple_des_suites_des_planches(
    terme: sp.Expr, point: object, limite: object
) -> None:
    """`2x`, `x`, `1 / (1 + x)`, puis les deux limites nulles des surprises 3 et 4."""
    assert limite_simple(terme, N, point) == limite


def test_la_limite_de_x_puissance_n_est_discontinue() -> None:
    """Page 7 : 0 sur `[0, 1[`, 1 en `x = 1`."""
    assert limite_simple(PUISSANCE, N, sp.Rational(9, 10)) == 0
    assert limite_simple(PUISSANCE, N, 1) == 1


def test_x_puissance_n_ne_converge_pas_uniformement() -> None:
    """Pages 13 et 19 : sur `[0, 1[`, `sup |x^n - 0|` reste egal a 1."""
    ouvert = sp.Interval.Ropen(0, 1)
    assert distances_uniformes(PUISSANCE, 0, N, ouvert, (1, 10, 100)) == (1, 1, 1)


def test_la_distance_uniforme_de_la_page_23_tend_vers_zero() -> None:
    """Page 23 : pour `sin(nx) / racine(n)`, `d(f, f_n) = 1 / racine(n)`."""
    periode = sp.Interval(0, 2 * sp.pi)
    terme = sp.sin(N * X) / sp.sqrt(N)
    attendues = (1, sp.Rational(1, 2), sp.Rational(1, 3))
    assert distances_uniformes(terme, 0, N, periode, (1, 4, 9)) == attendues


def test_la_convergence_uniforme_de_l_exemple_de_2025() -> None:
    """2025, page 24 : `|f_n - f| <= 1 / n` sur `[0, +oo[`."""
    terme, limite = N / (1 + N * (1 + X)), 1 / (1 + X)
    demi_droite = sp.Interval(0, sp.oo)
    rangs = (1, 2, 10)
    distances = distances_uniformes(terme, limite, N, demi_droite, rangs)
    assert all(d <= sp.Rational(1, n) for d, n in zip(distances, rangs, strict=True))


def test_le_critere_de_weierstrass_de_la_page_30() -> None:
    """Pages 29 et 30 : `cos(nx) / n3` est majoree par `1 / n3`, serie de Riemann."""
    periode = sp.Interval(0, 2 * sp.pi)
    for rang in (1, 2, 5):
        terme = sp.cos(rang * X) / rang**3
        assert est_majoree_par(terme, sp.Rational(1, rang**3), X, periode)
    trop_petit = sp.Rational(1, 9)
    assert not est_majoree_par(sp.cos(2 * X) / 8, trop_petit, X, periode)
    assert serie_majorante_convergente(1 / N**3, N)
