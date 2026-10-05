"""Notes de cours de 1993 (M. Pirlot), chapitre 4 : les suites de fonctions.

Source : `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, pages 92 a
105 des notes (pages 107 a 121 du fichier, un scan lu en image). Les
exemples des paragraphes 4.2 a 4.5 : convergence simple sans convergence
uniforme, le rang `N(eps, x)`, la distance `d(f_n, f)`, et les trois
contre-exemples qui montrent ou l'uniformite est indispensable.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.series.suites_de_fonctions import (
    distances_uniformes,
    limite_simple,
    rang_de_la_puissance,
    tente,
)
from polytech.maths.domain.symboles import VARIABLE as X

N = sp.Symbol("n", integer=True, positive=True)
R = sp.Rational
UNITE = sp.Interval(0, 1)


def test_la_limite_simple_des_puissances() -> None:
    """4.2 a) : `x^n` sur `[0, 1]` tend vers 0 pour `x < 1` et vers 1 en `x = 1`."""
    assert limite_simple(X**N, N, R(1, 2)) == 0
    assert limite_simple(X**N, N, R(9, 10)) == 0
    assert limite_simple(X**N, N, 1) == 1


def test_la_limite_simple_de_l_exemple_b() -> None:
    """4.2 b) : `1 / (1 + n^x)` tend vers `1/2` en 0 et vers 0 sur `]0, 1]`."""
    terme = 1 / (1 + N**X)
    assert limite_simple(terme, N, 0) == R(1, 2)
    assert limite_simple(terme, N, R(1, 2)) == 0
    assert limite_simple(terme, N, 1) == 0


def test_le_creneau_qui_glisse_tend_vers_zero_sans_s_en_approcher() -> None:
    """4.2 c) : 1 sur `[n, n + 1[`, 0 ailleurs. La limite est 0 partout ; `d = 1`."""
    creneau = sp.Piecewise((1, (X >= N) & (X < N + 1)), (0, True))
    for point in (0, R(7, 2), 100):
        assert limite_simple(creneau, N, point) == 0
    loin = sp.Interval(0, 1000)
    assert distances_uniformes(creneau, 0, N, loin, (1, 5, 50)) == (1, 1, 1)


def test_le_rang_des_puissances_depend_du_point() -> None:
    """Page 96 : `x^n < eps` ssi `n > ln(eps) / ln(x)`, et ce rang n'est pas borne."""
    rang = rang_de_la_puissance(R(1, 100), R(1, 2))
    assert rang == sp.log(100) / sp.log(2)
    assert R(1, 2) ** 7 < R(1, 100) <= R(1, 2) ** 6
    assert 6 < rang < 7
    x = sp.Symbol("x", positive=True)
    assert sp.limit(rang_de_la_puissance(R(1, 100), x), x, 1, "-") == sp.oo


def test_le_rang_n_a_de_sens_que_pour_un_point_et_un_seuil_entre_zero_et_un() -> None:
    with pytest.raises(ValueError, match="strictement entre 0 et 1"):
        rang_de_la_puissance(R(1, 100), 1)


def test_les_puissances_ne_convergent_pas_uniformement() -> None:
    """Page 97, exemple 2 : `d(x^n, f) = sup_{0 <= x < 1} x^n = 1` pour tout `n`."""
    ouvert = sp.Interval.Ropen(0, 1)
    assert distances_uniformes(X**N, 0, N, ouvert, (1, 3, 20)) == (1, 1, 1)


def test_les_puissances_convergent_uniformement_loin_de_un() -> None:
    """Sur `[0, 1/2]` la distance vaut `2^(-n)`."""
    moitie = sp.Interval(0, R(1, 2))
    attendues = (R(1, 2), R(1, 8), R(1, 1024))
    assert distances_uniformes(X**N, 0, N, moitie, (1, 3, 10)) == attendues


def test_la_tente_de_la_page_95() -> None:
    """Hauteur `n` en `1/n`, nulle hors de `[0, 2/n]` : aire 1 quel que soit `n`."""
    for rang in (1, 2, 5):
        fonction = tente(rang)
        assert fonction.subs(X, R(1, rang)) == rang
        assert fonction.subs(X, 0) == 0
        assert fonction.subs(X, R(2, rang)) == 0
        assert fonction.subs(X, 3) == 0
        assert sp.integrate(fonction, (X, 0, 2)) == 1


def test_la_tente_tend_vers_zero_en_chaque_point_mais_d_vaut_n() -> None:
    """Page 97, exemple 1 : `f_n(x) = 0` des que `n > 2/x`, et `d(f_n, 0) = n`."""
    for point in (R(1, 2), R(1, 10)):
        suffisant = int(2 / point) + 1
        assert tente(suffisant).subs(X, point) == 0
    distances = tuple(
        distances_uniformes(tente(rang), 0, N, sp.Interval(0, 2), (rang,))[0]
        for rang in (1, 4, 9)
    )
    assert distances == (1, 4, 9)


def test_sans_convergence_uniforme_l_integrale_ne_passe_pas_a_la_limite() -> None:
    """Theoreme 4.5.2, paraphrase : `lim int f_n = 1` alors que `int lim f_n = 0`."""
    integrales = [sp.integrate(tente(rang), (X, 0, 2)) for rang in (1, 3, 8)]
    assert integrales == [1, 1, 1]
    assert sp.integrate(sp.Integer(0), (X, 0, 2)) == 0


def test_sur_un_intervalle_infini_meme_la_convergence_uniforme_ne_suffit_pas() -> None:
    """Page 100 : `1/n` sur `]-n, n[`. `d = 1/n` tend vers 0, mais `int f_n = 2`."""
    plateau = sp.Piecewise((1 / N, sp.Abs(X) < N), (0, True))
    large = sp.Interval(-100, 100)
    assert distances_uniformes(plateau, 0, N, large, (1, 4, 10)) == (
        1,
        R(1, 4),
        R(1, 10),
    )
    for rang in (1, 4, 10):
        aire = sp.integrate(plateau.subs(N, rang), (X, -sp.oo, sp.oo))
        assert aire == 2


def test_la_derivee_ne_passe_pas_a_la_limite_sans_hypothese_sur_les_derivees() -> None:
    """Page 101 : `sin(nx) / racine(n)` tend uniformement vers 0 ; `f_n'(0)` diverge."""
    terme = sp.sin(N * X) / sp.sqrt(N)
    periode = sp.Interval(0, 2 * sp.pi)
    attendues = (1, R(1, 2), R(1, 3))
    assert distances_uniformes(terme, 0, N, periode, (1, 4, 9)) == attendues
    derivee_en_zero = sp.diff(terme, X).subs(X, 0)
    assert derivee_en_zero == sp.sqrt(N)
    assert sp.limit(derivee_en_zero, N, sp.oo) == sp.oo


def test_une_limite_discontinue_trahit_une_convergence_non_uniforme() -> None:
    """Theoreme 4.5.1 : la limite uniforme de fonctions continues est continue."""
    gauche = limite_simple(X**N, N, R(999, 1000))
    assert gauche != limite_simple(X**N, N, 1)
    assert distances_uniformes(X**N, 0, N, sp.Interval.Ropen(0, 1), (50,)) == (1,)
