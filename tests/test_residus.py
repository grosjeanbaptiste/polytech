"""Le residu d'une fonction en une singularite isolee, et le theoreme des residus.

Oracles : `MathematiquesIngenieur3/SYLLABUS/Syllabus_Feuillet_1_Analyse_complexe.pdf`,
section 9, pages 26 a 29 (lues sur la page rendue).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.analyse_complexe.integrales_reelles import (
    integrale_de_1_sur_a_plus_cosinus,
    poles_de_1_sur_a_plus_cosinus,
    quadrature_de_1_sur_a_plus_cosinus,
)
from polytech.maths.domain.analyse_complexe.residus import (
    integrale_par_les_residus,
    limite_du_produit,
    residu,
    residu_d_un_pole,
    residu_d_un_pole_simple,
)
from polytech.maths.domain.analyse_complexe.singularites import ordre_du_pole

Z = sp.Symbol("z")
ALPHA, BETA = sp.symbols("alpha beta")
DEUX_POLES = -sp.I / ((Z - ALPHA) * (Z - BETA))


def test_le_residu_d_un_pole_simple_est_la_limite_du_produit() -> None:
    """Page 29, methode 2 : `-i / (alpha - beta)` en `alpha`."""
    obtenu = residu_d_un_pole_simple(DEUX_POLES, Z, ALPHA)
    assert sp.simplify(obtenu + sp.I / (ALPHA - BETA)) == 0


def test_une_limite_nulle_dit_une_singularite_reguliere() -> None:
    """Page 27, methodes 1 et 2 : limite nulle, residu nul."""
    assert residu_d_un_pole_simple(sp.sin(Z) / Z, Z, 0) == 0


def test_un_pole_double_n_est_pas_un_pole_simple() -> None:
    """Page 27, methode 2 : la limite doit exister et ne pas etre infinie."""
    with pytest.raises(ValueError, match="pole simple"):
        residu_d_un_pole_simple(1 / Z**2, Z, 0)


def test_la_limite_du_produit_ne_juge_pas() -> None:
    """Le produit `f(z)(z - a)` seul : il diverge sur un pole multiple."""
    assert limite_du_produit(1 / Z**2, Z, 0) == sp.oo


@pytest.mark.parametrize("n", [2, 3, 5])
def test_le_residu_d_un_pole_d_ordre_h(n: int) -> None:
    """Page 28, (115) : `Res((e^z - 1) / z^n, 0) = 1 / (n - 1)!`.

    Methode 3, page 27 : `g = f (z - a)^h`, `Res = g^(h-1)(a) / (h - 1)!`.
    """
    fonction = (sp.exp(Z) - 1) / Z**n
    attendu = 1 / sp.factorial(n - 1)
    assert residu_d_un_pole(fonction, Z, 0, ordre=n) == attendu
    assert residu(fonction, Z, 0) == attendu


def test_un_ordre_nul_est_refuse() -> None:
    """La methode 3 demande `h >= 1`."""
    with pytest.raises(ValueError, match="ordre"):
        residu_d_un_pole(1 / Z, Z, 0, ordre=0)


@pytest.mark.parametrize(
    ("fonction", "ordre"),
    [
        (sp.sin(Z) / Z, 0),
        (sp.sin(Z), 0),
        (1 / Z, 1),
        (sp.exp(Z) / Z**3, 3),
        ((sp.exp(Z) - 1) / Z**3, 2),
    ],
)
def test_l_ordre_du_pole(fonction: sp.Expr, ordre: int) -> None:
    """Page 27 : le plus petit `h` tel que `f (z - a)^h` ait une limite finie."""
    assert ordre_du_pole(fonction, Z, 0) == ordre


def test_le_residu_en_une_singularite_reguliere_est_nul() -> None:
    """Page 27, methode 1 : singularite reguliere, residu nul."""
    assert residu(sp.sin(Z) / Z, Z, 0) == 0


def test_une_singularite_essentielle_n_a_pas_d_ordre() -> None:
    """Page 27 : les methodes 1 a 4 ne valent pas pour une singularite essentielle."""
    with pytest.raises(ValueError, match="essentielle"):
        ordre_du_pole(sp.exp(1 / Z), Z, 0)


def test_le_residu_d_une_singularite_essentielle_passe_par_laurent() -> None:
    """Page 29 : `Res(sin(1/z), 0) = 1`."""
    assert residu(sp.sin(1 / Z), Z, 0) == 1


def test_l_integrale_de_l_exemple_de_la_page_26_vaut_deux_pi_sur_racine() -> None:
    """Page 26 : `I = int de 0 a 2 pi de d(theta) / (a + cos(theta))`, `a > 1`.

    Ecart au cours : le syllabus conclut `pi / racine(a2 - 1)`, relation (112).
    Il factorise `z2/2 + a z + 1/2` en `(z - alpha)(z - beta)` : le facteur
    `1/2` est perdu. Avec `f = -2i / ((z - alpha)(z - beta))`, le theoreme 16
    donne `2 pi / racine(a2 - 1)`. Second chemin, pour `a = 2` : la
    quadrature de l'integrale reelle.
    """
    a = 2
    alpha, beta = poles_de_1_sur_a_plus_cosinus(a)
    assert sp.Abs(alpha) < 1 < sp.Abs(beta)
    integrale = integrale_de_1_sur_a_plus_cosinus(a)
    facteurs = -2 * sp.I / ((Z - alpha) * (Z - beta))
    factorisee = integrale_par_les_residus(facteurs, Z, (alpha,))
    assert sp.simplify(integrale - factorisee) == 0
    imprimee = integrale_par_les_residus(
        -sp.I / ((Z - alpha) * (Z - beta)), Z, (alpha,)
    )
    assert sp.simplify(integrale - 2 * imprimee) == 0
    assert sp.simplify(integrale - imprimee) != 0
    assert float(integrale) == pytest.approx(
        quadrature_de_1_sur_a_plus_cosinus(a), rel=1e-12
    )
    with pytest.raises(ValueError, match="a > 1"):
        integrale_de_1_sur_a_plus_cosinus(1)
