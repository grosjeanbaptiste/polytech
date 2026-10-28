"""Fiche de signaux : second ordre et linearisation.

Articles « second ordre », « trois regimes » et « lineariser » de
`cursus/fiches/SignauxSystemes.toml`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.comparaison import egales, proches
from polytech.signaux.domain.reponse_indicielle import (
    classe_d_amortissement,
    depassement_du_second_ordre,
)
from polytech.signaux.domain.reponses_du_second_ordre import (
    indicielle,
    instant_du_maximum,
)
from polytech.signaux.domain.second_ordre import SecondOrdre
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import TEMPS as T
from polytech.signaux.domain.systemes import (
    linearisation_de_taylor,
    point_de_fonctionnement,
)

R, C = sp.symbols("R C", positive=True)
A, B = sp.symbols("a b", positive=True)
H_ = sp.Symbol("h", positive=True)


def test_poles_et_pulsation_du_second_ordre() -> None:
    """Article « Le second ordre » : `p = (-xi +- sqrt(xi^2-1)) / T`."""
    xi, t0 = sp.Rational(3, 5), sp.Integer(2)
    systeme = SecondOrdre(gain=1, constante=t0, amortissement=xi)
    attendu = {(-xi + s * sp.sqrt(xi**2 - 1)) / t0 for s in (1, -1)}
    assert {sp.simplify(p) for p in systeme.poles()} == {
        sp.simplify(p) for p in attendu
    }
    assert egales(abs(sp.im(systeme.poles()[0])), sp.sqrt(1 - xi**2) / t0)
    assert SecondOrdre(gain=1, constante=1, amortissement=1).regime() == (
        "poles reels confondus"
    )


def test_la_frontiere_sans_oscillation_est_zeta_un() -> None:
    """Juin 2016 (2) et juin 2026 (2) : `p^2 + 2000 p + 10^6 = (p + 1000)^2`."""
    critique = SecondOrdre(gain=1, constante=sp.Rational(1, 1000), amortissement=1)
    assert set(critique.poles()) == {-1000}
    assert classe_d_amortissement(1.0) != classe_d_amortissement(0.3)
    assert egales(sp.expand((P + 1000) ** 2), P**2 + 2000 * P + 10**6)
    h = SecondOrdre(gain=1, constante=1, amortissement=sp.Rational(1, 2)).de_bode()
    assert sp.limit(P * h.expression, P, sp.oo) == 0


def test_le_depassement_est_l_exces_du_maximum_de_la_reponse() -> None:
    """Article « Les trois regimes » : `D = exp(-pi xi / sqrt(1 - xi^2))`."""
    xi = sp.Rational(3, 10)
    systeme = SecondOrdre(gain=1, constante=1, amortissement=xi)
    maximum = indicielle(systeme).subs(T, instant_du_maximum(systeme))
    assert proches(float(maximum - 1), float(depassement_du_second_ordre(xi)), 1e-9)


def test_la_linearisation_est_la_tangente_au_point_de_repos() -> None:
    """Article « Lineariser » : `q = sqrt(h)` en `h = 2`, pente `1/(2 sqrt 2)`."""
    tangente = linearisation_de_taylor(sp.sqrt(H_), H_, 2)
    assert egales(tangente, sp.sqrt(2) + (H_ - 2) / (2 * sp.sqrt(2)))
    u = sp.Symbol("u", positive=True)
    assert point_de_fonctionnement(u - sp.sqrt(H_), H_) == (u**2,)


def test_les_poles_de_la_figure_sont_sur_le_cercle_de_rayon_omega_n() -> None:
    """Figure : poles a `135` degres, `|p| = omega_n`, `zeta = -cos(theta)`."""
    theta = sp.Rational(3, 4) * sp.pi
    pole = 2 * (sp.cos(theta) + sp.I * sp.sin(theta))
    systeme = SecondOrdre(
        gain=1, constante=sp.Rational(1, 2), amortissement=-sp.cos(theta)
    )
    assert any(egales(sp.expand(p - pole), 0) for p in systeme.poles())
    assert all(egales(sp.Abs(p), 2) for p in systeme.poles())


def test_les_courbes_des_trois_regimes_de_la_figure() -> None:
    """Figure « Les trois regimes » : `zeta = 0,3`, critique, `zeta = 1,5`."""
    for x in (0.5, 2.0, 7.0):
        sous = 1 - sp.exp(-0.3 * x) * (
            sp.cos(x * sp.sqrt(0.91)) + 0.3 / sp.sqrt(0.91) * sp.sin(x * sp.sqrt(0.91))
        )
        critique = 1 - (1 + x) * sp.exp(-x)
        racine = sp.sqrt(1.25)
        sur = 1 - (
            (1.5 + racine) * sp.exp(-(1.5 - racine) * x)
            - (1.5 - racine) * sp.exp(-(1.5 + racine) * x)
        ) / (2 * racine)
        for zeta, figure in (
            (sp.Rational(3, 10), sous),
            (sp.Integer(1), critique),
            (sp.Rational(3, 2), sur),
        ):
            systeme = SecondOrdre(gain=1, constante=1, amortissement=zeta)
            lib = sp.N(indicielle(systeme).subs(T, x))
            assert proches(float(lib), float(figure), 1e-9)
