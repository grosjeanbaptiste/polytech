"""Les trois documents manuscrits de mathematique rapatries le 2026-10-21.

Sources, dans `Mathematique/` :

- `EXS/Exs_02_Equations_differentielles_Notes_de_seance_suite.pdf` -- la
  suite des notes de seance du chapitre 2 (huit pages) ;
- `EXS/Exs_03_Transformee_de_Laplace_Notes_de_seance.pdf` -- l'enonce
  imprime des exercices sur Laplace et ses notes (quatre pages) ;
- `SLIDES/Slides_03_Transformee_de_Laplace_annote.pdf` -- le fascicule 3
  annote a la main sur vingt-deux de ses quarante-deux planches.

Transcrits a la main dans `transcriptions/Mathematique/`. Comme pour les
notes des chapitres 1 et 2, aucune ligne n'est recopiee pour etre comparee
a elle-meme : une solution est reportee dans son equation, une transformee
est recalculee par `sympy.laplace_transform` ou par quadrature. Le recueil
`EXS/Exs_02_Equations_differentielles.pdf` sert de second temoin la ou il
imprime une reponse.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import sympy as sp
from scipy import integrate

RACINE = Path(__file__).resolve().parent.parent
TRANSCRIPTIONS = RACINE / "transcriptions" / "Mathematique"
T, X = sp.symbols("t x", real=True)
P, OMEGA = sp.symbols("p omega", positive=True)
A, PHI, C, K = sp.symbols("a phi C K", real=True)
Y = sp.Function("y")


def _laplace(expression: sp.Expr) -> sp.Expr:
    return sp.laplace_transform(expression, T, P, noconds=True)


# --- Suite des notes du chapitre 2 ------------------------------------------


def test_exercice_7_la_solution_verifie_l_equation() -> None:
    """Page 1 : `y = (x + C) cos x` pour `y' + y tg x = cos x`."""
    y = (X + C) * sp.cos(X)
    assert sp.simplify(sp.diff(y, X) + y * sp.tan(X) - sp.cos(X)) == 0


def test_exercice_7_le_recueil_imprime_la_meme_famille() -> None:
    """Le recueil, exercice I.1.6 : `y = (K + x) cos x` ; dsolve le redonne."""
    solution = sp.dsolve(sp.Eq(Y(X).diff(X) + Y(X) * sp.tan(X), sp.cos(X)), Y(X)).rhs
    constante = next(iter(solution.free_symbols - {X}))
    assert sp.simplify(solution.subs(constante, K) - (K + X) * sp.cos(X)) == 0


def test_l_equation_homogene_donne_k_cos_x() -> None:
    """Page 2 : `int tg = -ln|cos|`, donc `y_SGEH = K cos x`."""
    assert sp.simplify(sp.diff(-sp.log(sp.cos(X)), X) - sp.tan(X)) == 0
    y = K * sp.cos(X)
    assert sp.simplify(sp.diff(y, X) + y * sp.tan(X)) == 0


def test_la_variation_de_la_constante_donne_k_egal_x() -> None:
    """Page 3 : `K' cos x = cos x`, donc `K = x` et `y_SPENH = x cos x`."""
    k = sp.Function("K")
    y = k(X) * sp.cos(X)
    reste = sp.simplify(sp.diff(y, X) + y * sp.tan(X) - sp.cos(X))
    assert sp.simplify(reste - (k(X).diff(X) - 1) * sp.cos(X)) == 0


def test_l_equation_exacte_demande_alpha_egal_1() -> None:
    """Pages 4 et 5, et la reponse du recueil : `alpha = 1`."""
    alpha, y = sp.symbols("alpha y", real=True)
    m = 2 * X + alpha * y + 2 * X * sp.exp(X**2)
    n = 2 * y + X
    assert sp.solve(sp.Eq(sp.diff(m, y), sp.diff(n, X)), alpha) == [1]


def test_la_fonction_potentielle_de_l_equation_exacte() -> None:
    """Page 5 : `G = x^2 + yx + e^(x^2) + y^2`, ses derivees sont M et N."""
    y = sp.Symbol("y", real=True)
    g = X**2 + y * X + sp.exp(X**2) + y**2
    assert sp.simplify(sp.diff(g, X) - (2 * X + y + 2 * X * sp.exp(X**2))) == 0
    assert sp.simplify(sp.diff(g, y) - (2 * y + X)) == 0


def test_la_famille_de_courbes_verifie_f_prime_egal_x_sur_f() -> None:
    """Pages 7 et 8 : `f' = x / f`, dont la famille est `y^2 = x^2 + K`."""
    y = sp.sqrt(X**2 + K)
    assert sp.simplify(sp.diff(y, X) - X / y) == 0


@pytest.mark.parametrize(("x", "k"), [(1.0, 3.0), (2.0, -1.5), (0.5, 0.75)])
def test_sur_la_famille_les_angles_sont_complementaires(x: float, k: float) -> None:
    """Le probleme lui-meme : `alpha + beta = pi / 2` en tout point."""
    y = math.sqrt(x**2 + k)
    beta = math.atan(x / y)
    alpha = math.atan(y / x)
    assert alpha + beta == pytest.approx(math.pi / 2)


# --- Notes sur la transformee de Laplace ------------------------------------


def test_cos_dephase_par_linearite() -> None:
    """Page 1 et 2, 1 d) : `cos(phi) p/(p^2+w^2) - sin(phi) w/(p^2+w^2)`."""
    attendu = (sp.cos(PHI) * P - sp.sin(PHI) * OMEGA) / (P**2 + OMEGA**2)
    direct = _laplace(sp.cos(OMEGA * T + PHI))
    assert sp.simplify(direct - attendu) == 0


def test_la_decomposition_du_1_b_par_linearite() -> None:
    """1 b) : la transformee du tout est la somme des trois transformees."""
    tout = _laplace(sp.exp(-2 * T) * (sp.cos(T) + 2 * T**2 - 1))
    parties = (
        _laplace(sp.exp(-2 * T) * sp.cos(T))
        + 2 * _laplace(sp.exp(-2 * T) * T**2)
        - _laplace(sp.exp(-2 * T))
    )
    assert sp.simplify(tout - parties) == 0
    sans_le_facteur_2 = parties - _laplace(sp.exp(-2 * T) * T**2)
    assert sp.simplify(tout - sans_le_facteur_2) != 0  # l'ecart des notes


def test_les_deux_fonctions_du_2_ne_different_qu_entre_0_et_1() -> None:
    """Page 2 : `f = (t-1)^3 H(t)` vaut -1 en 0, `g = tau_1(H t^3)` y est nulle."""
    f = (T - 1) ** 3 * sp.Heaviside(T, 1)  # planche 11 : H(0) = 1
    g = (T - 1) ** 3 * sp.Heaviside(T - 1, 1)
    assert f.subs(T, 0) == -1
    assert g.subs(T, sp.Rational(1, 2)) == 0
    assert sp.simplify((f - g).subs(T, 3)) == 0


@pytest.mark.parametrize("p", [0.7, 1.3, 4.0])
def test_la_dent_de_scie_du_3_a(p: float) -> None:
    """Page 3 : `e^(-p) / (-p) . 1 / (1 - e^(-p)) + 1 / p^2`, par quadrature."""
    formule = math.exp(-p) / (-p) / (1 - math.exp(-p)) + 1 / p**2
    periodes = range(60)
    quadrature = sum(
        integrate.quad(lambda t, n=n: math.exp(-p * t) * (t - n), n, n + 1)[0]
        for n in periodes
    )
    assert quadrature == pytest.approx(formule, rel=1e-9)


# --- Les annotations du fascicule 3 -----------------------------------------


def test_planche_6_la_derivee_d_une_serie_entiere() -> None:
    """`y' = sum n a_n x^(n-1)` : la derivee decale les coefficients."""
    n = 6
    a = sp.symbols(f"a0:{n + 1}")
    serie = sum(a[k] * X**k for k in range(n + 1))
    decalee = sum(k * a[k] * X ** (k - 1) for k in range(1, n + 1))
    assert sp.expand(sp.diff(serie, X) - decalee) == 0


def test_planche_6_la_somme_partielle_de_la_serie_geometrique() -> None:
    """`sum_0^n x^k = (1 - x^(n+1)) / (1 - x)`, et `x^(n+1) -> 0` si `|x| < 1`."""
    for n in range(8):
        somme = sum(X**k for k in range(n + 1))
        assert sp.cancel(somme - (1 - X ** (n + 1)) / (1 - X)) == 0
    n = sp.Symbol("n", integer=True, nonnegative=True)
    assert sp.limit(sp.Rational(1, 2) ** n, n, sp.oo) == 0


def test_planche_7_x_puissance_t_est_e_moins_s_t() -> None:
    """`s = -ln x`, donc `x^t = e^(-s t)`."""
    x = sp.Symbol("x", positive=True)
    s = -sp.log(x)
    assert sp.simplify(sp.exp(-s * T) - x**T) == 0


def test_planche_10_l_integrale_ne_converge_que_pour_re_p_positif() -> None:
    """L'imprime ecrit `Re(p) < 0`, l'encre corrige en `> 0`."""
    positif = sp.Symbol("s", positive=True)
    negatif = sp.Symbol("s", negative=True)
    assert sp.integrate(sp.exp(-positif * T), (T, 0, sp.oo)) == 1 / positif
    assert sp.integrate(sp.exp(-negatif * T), (T, 0, sp.oo)) == sp.oo


@pytest.mark.parametrize("x", [sp.sin(3 * T), sp.exp(2 * T) + T**2, sp.cos(T) - 4])
def test_planches_12_et_14_les_derivees_premiere_et_seconde(x: sp.Expr) -> None:
    """`L(x') = pX - x(0)` et `L(x'') = p^2 X - p x(0) - x'(0)`."""
    grand_x = _laplace(x)
    x0 = x.subs(T, 0)
    xp0 = sp.diff(x, T).subs(T, 0)
    assert sp.simplify(_laplace(sp.diff(x, T)) - (P * grand_x - x0)) == 0
    seconde = P**2 * grand_x - P * x0 - xp0
    assert sp.simplify(_laplace(sp.diff(x, T, 2)) - seconde) == 0


def test_planche_16_l_inverse_resout_l_equation() -> None:
    """`X(p) = 1 / (p (p^2 + 3p + 2))` : l'original verifie `x'' + 3x' + 2x = 1`."""
    t = sp.Symbol("t", positive=True)
    x = sp.inverse_laplace_transform(1 / (P * (P**2 + 3 * P + 2)), P, t)
    residu = sp.diff(x, t, 2) + 3 * sp.diff(x, t) + 2 * x - 1
    assert sp.simplify(residu) == 0
    assert sp.simplify(x.subs(t, 0)) == 0


def test_planche_20_un_pole_dans_l_intervalle_n_est_pas_integrable() -> None:
    """`x(t) = 1 / (t - A)` : l'integrale de `|x|` diverge pres de `A`."""
    epsilon = sp.Symbol("epsilon", positive=True)
    jusque_pres_du_pole = sp.integrate(1 / (1 - T), (T, 0, 1 - epsilon))
    assert sp.limit(jusque_pres_du_pole, epsilon, 0, "+") == sp.oo


def test_planches_25_et_26_l_exponentielle() -> None:
    """`X(p) = 1 / (p - a)` pour `Re(p) > a` ; la limite s'annule alors."""
    a = sp.Symbol("a", positive=True)
    assert sp.simplify(_laplace(sp.exp(a * T)) - 1 / (P - a)) == 0
    ecart, beta = sp.symbols("ecart beta", positive=True)
    assert sp.limit(sp.exp(-ecart * beta), beta, sp.oo) == 0


def test_planche_27_la_primitive_du_sinus_amorti() -> None:
    """`[e^(-pt) (-w cos wt - p sin wt) / (p^2 + w^2)]` et sa valeur `w/(p^2+w^2)`."""
    primitive = (
        sp.exp(-P * T)
        * (-OMEGA * sp.cos(OMEGA * T) - P * sp.sin(OMEGA * T))
        / (P**2 + OMEGA**2)
    )
    integrande = sp.exp(-P * T) * sp.sin(OMEGA * T)
    assert sp.simplify(sp.diff(primitive, T) - integrande) == 0
    valeur = sp.limit(primitive, T, sp.oo) - primitive.subs(T, 0)
    assert sp.simplify(valeur - OMEGA / (P**2 + OMEGA**2)) == 0


@pytest.mark.parametrize("n", range(1, 7))
def test_planches_32_et_33_la_recurrence_des_puissances(n: int) -> None:
    """`X_n = (n / p) X_(n-1)`, `X_0 = 1/p`, donc `X_n = n! / p^(n+1)`."""
    precedent = _laplace(T ** (n - 1))
    courant = _laplace(T**n)
    assert sp.simplify(courant - n / P * precedent) == 0
    assert sp.simplify(courant - sp.factorial(n) / P ** (n + 1)) == 0


# --- Les transcriptions elles-memes -----------------------------------------


@pytest.mark.parametrize(
    ("fichier", "resultats"),
    [
        (
            "EXS/Exs_02_Equations_differentielles_Notes_de_seance_suite.md",
            (r"(x + C)\cos(x)", r"\alpha = 1", r"y^2 = x^2 + K"),
        ),
        (
            "EXS/Exs_03_Transformee_de_Laplace_Notes_de_seance.md",
            (r"\frac{1}{1 - e^{-p}} + \frac{1}{p^2}", r"\frac{\omega}{p^2 + \omega^2}"),
        ),
        (
            "SLIDES/Slides_03_Transformee_de_Laplace_annote.md",
            (r"X_n(p) = \frac{n!}{p^{n+1}}", r"\Re(p) > 0", r"x^t = e^{-st}"),
        ),
    ],
)
def test_la_transcription_porte_les_resultats_verifies(
    fichier: str, resultats: tuple[str, ...]
) -> None:
    """Ce que ce fichier recalcule est bien ce que la transcription ecrit."""
    texte = (TRANSCRIPTIONS / fichier).read_text(encoding="utf-8")
    assert texte.startswith("<!--\nTranscription visuelle d'un scan manuscrit")
    assert [r for r in resultats if r not in texte] == []
