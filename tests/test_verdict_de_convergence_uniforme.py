"""Le verdict de convergence uniforme : `d(f_n, f) -> 0` ou non.

Source : `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, pages 111
et 115 (la suite `x^n`, la suite `sin(nx) / racine(n)`). Les distances
attendues sont celles de ces pages ; le verdict recoupe la distance annoncee
avec le plus grand ecart calcule sur des rangs, avant de passer a la limite.
"""

import pytest
import sympy as sp

from polytech.maths.domain.series.verdict_uniforme import Suite, Verdict, verdict

X = sp.Symbol("x", real=True)
N = sp.Symbol("n", positive=True, integer=True)
DEMI = sp.Rational(1, 2)


def test_x_puissance_n_converge_uniformement_sur_zero_a() -> None:
    d = DEMI**N
    assert (
        verdict(Suite(X**N, 0, N, sp.Interval(0, DEMI)), d, (1, 2, 3))
        is Verdict.UNIFORME
    )


def test_x_puissance_n_ne_converge_pas_uniformement_sur_zero_un_ouvert() -> None:
    intervalle = sp.Interval.Ropen(0, 1)
    resultat = verdict(Suite(X**N, 0, N, intervalle), sp.Integer(1), (1, 2, 3))
    assert resultat is Verdict.PAS_UNIFORME


def test_sin_nx_sur_racine_de_n_converge_uniformement() -> None:
    d = 1 / sp.sqrt(N)
    rangs = (2, 3, 4)
    terme = sp.sin(N * X) / sp.sqrt(N)
    assert verdict(Suite(terme, 0, N, sp.Interval(0, 1)), d, rangs) is Verdict.UNIFORME


def test_une_distance_annoncee_fausse_est_refusee() -> None:
    with pytest.raises(ValueError, match="ne vaut pas le plus grand ecart"):
        verdict(Suite(X**N, 0, N, sp.Interval(0, DEMI)), DEMI ** (N + 1), (1, 2, 3))


def test_une_limite_indeterminee_est_refusee() -> None:
    d = sp.sin(N) ** 2 * 0 + sp.Abs(sp.sin(N))
    terme = sp.Abs(sp.sin(N)) * X
    with pytest.raises(ValueError, match="limite de la distance indeterminee"):
        verdict(Suite(terme, 0, N, sp.Interval(0, 1)), d, (1, 2, 3))


def test_une_distance_qui_tend_vers_deux_n_est_pas_uniforme() -> None:
    suite = Suite(2 * X**N, 0, N, sp.Interval.Ropen(0, 1))
    assert verdict(suite, sp.Integer(2), (1, 2, 3)) is Verdict.PAS_UNIFORME


def test_la_tente_de_la_page_111_a_une_distance_qui_diverge() -> None:
    """`n x` sur `[0, 1]` : `d_n = n`, la distance tend vers l'infini."""
    suite = Suite(N * X, 0, N, sp.Interval(0, 1))
    assert verdict(suite, N, (1, 2, 3)) is Verdict.PAS_UNIFORME
