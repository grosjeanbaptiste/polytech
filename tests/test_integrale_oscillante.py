"""L'integrale d'une rationnelle fois une exponentielle oscillante, par les residus.

Oracle : planches du calcul d'integrales,
`MathematiquesIngenieur3/SLIDES/Slides_09_Calcul_des_residus_partie_2_Calcul_d_integrales.pdf`,
planches 19 a 22 : `I = int_R R(x) e^(ix) dx`, `R = p / q`, trois cas --
(a) `q` sans zero reel et `deg q >= deg p + 2` ; (b) `deg q = deg p + 1`,
par un rectangle ; (c) des zeros reels simples, en valeur principale,
`2 pi i somme_(Im > 0) Res + i pi somme_(Im = 0) Res` ; et l'exemple
`int sin x / x = pi`, planche 22. Second chemin : la quadrature oscillante
de mpmath. Les examens : examen blanc du 3 decembre 2025, question 2
(`x^2 cos x / (1 + x^4)`) ; janvier 2026, question 2 b (la transformee de
Fourier de `1 / (1 + x^2)^2`, `e^(-isx)` avec `s` des deux signes).
"""

from __future__ import annotations

import math

import mpmath
import pytest
import sympy as sp

from polytech.maths.domain.analyse_complexe.jordan import integrale_oscillante

x = sp.Symbol("x", real=True)


def _quadrature(rationnelle: sp.Expr, frequence: float) -> complex:
    reel = sp.lambdify(x, rationnelle, "mpmath")
    periode = 2 * mpmath.pi / abs(frequence)
    cos = mpmath.quadosc(
        lambda u: reel(u) * mpmath.cos(frequence * u),
        [-mpmath.inf, mpmath.inf],
        period=periode,
    )
    sin = mpmath.quadosc(
        lambda u: reel(u) * mpmath.sin(frequence * u),
        [-mpmath.inf, mpmath.inf],
        period=periode,
    )
    return complex(cos + 1j * sin)


def test_sinus_cardinal_en_valeur_principale() -> None:
    """Planche 22 : `int sin x / x = pi`, donc `VP int e^(ix) / x = i pi`."""
    assert integrale_oscillante(1 / x, x, 1) == sp.I * sp.pi


def test_l_examen_blanc_cos_x_x_carre_sur_un_plus_x_quatre() -> None:
    """Cas (a) : la partie reelle de `int x^2 e^(ix) / (1 + x^4)`."""
    valeur = complex(integrale_oscillante(x**2 / (1 + x**4), x, 1))
    attendue = _quadrature(x**2 / (1 + x**4), 1.0)
    assert math.isclose(valeur.real, attendue.real, rel_tol=1e-10)
    assert abs(valeur.imag) < 1e-12


@pytest.mark.parametrize("s", [2, -2, sp.Rational(1, 3)])
def test_la_transformee_de_janvier_2026_des_deux_cotes(s: sp.Expr) -> None:
    """`int e^(-isx) / (1 + x^2)^2 = pi (1 + |s|) e^(-|s|) / 2`, poles doubles."""
    valeur = integrale_oscillante(1 / (1 + x**2) ** 2, x, -s)
    attendue = sp.pi * (1 + abs(s)) * sp.exp(-abs(s)) / 2
    assert sp.simplify(valeur - attendue) == 0


def test_le_cas_b_par_le_rectangle() -> None:
    """`deg q = deg p + 1` : `int x e^(ix) / (x^2 + 1) = i pi / e`."""
    valeur = integrale_oscillante(x / (x**2 + 1), x, 1)
    assert sp.simplify(valeur - sp.I * sp.pi / sp.E) == 0
    assert math.isclose(
        complex(valeur).imag, _quadrature(x / (x**2 + 1), 1.0).imag, rel_tol=1e-8
    )


def test_une_rationnelle_qui_ne_s_eteint_pas_est_refusee() -> None:
    with pytest.raises(ValueError, match="degre"):
        integrale_oscillante(x**2 / (1 + x**2), x, 1)


def test_une_frequence_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="frequence"):
        integrale_oscillante(1 / (1 + x**2), x, 0)


def test_un_pole_reel_double_est_refuse() -> None:
    """Le cas (c) ne vaut que pour des zeros reels simples."""
    with pytest.raises(ValueError, match="pole reel multiple"):
        integrale_oscillante(1 / x**2, x, 1)


def test_des_zeros_que_sympy_n_exprime_pas_levent() -> None:
    """`x^5 - x + 1` n'est pas resoluble par radicaux : aucune somme partielle."""
    with pytest.raises(ValueError, match="hors de portee"):
        integrale_oscillante(1 / (x**5 - x + 1), x, 1)
