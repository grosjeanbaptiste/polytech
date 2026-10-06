"""Le residu d'un quotient `g / h`, par l'inverse d'une serie de puissances.

Oracles : `MathematiquesIngenieur3/SYLLABUS/Syllabus_Feuillet_1_Analyse_complexe.pdf`,
section 9.2, methode 4, pages 27 a 29 (lues sur la page rendue).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.analyse_complexe.quotient import (
    inverse_d_une_serie,
    ordre_du_zero,
    residu_d_un_quotient,
)
from polytech.maths.domain.analyse_complexe.residus import residu_d_un_pole

Z = sp.Symbol("z")
ALPHA, BETA = sp.symbols("alpha beta")


def test_l_inverse_de_un_moins_z_est_la_serie_geometrique() -> None:
    """Page 27, (114), et page 28, d) : `(1 - z)^-1 = somme z^n`."""
    assert inverse_d_une_serie((1, -1, 0, 0, 0)) == (1, 1, 1, 1, 1)


def test_la_recurrence_de_l_inverse_sur_deux_termes() -> None:
    """Page 28, cas `m = 2` : `d_0 = 1 / b_2`, `d_1 = -b_3 / b_2^2`."""
    b2, b3 = sp.symbols("b2 b3", nonzero=True)
    d0, d1 = inverse_d_une_serie((b2, b3))
    assert (d0, sp.simplify(d1 + b3 / b2**2)) == (1 / b2, 0)


def test_une_serie_de_terme_constant_nul_n_a_pas_d_inverse() -> None:
    """Page 27 : la recurrence (114) demande `b_0` non nul."""
    with pytest.raises(ValueError, match="terme constant nul"):
        inverse_d_une_serie((0, 1))


@pytest.mark.parametrize(
    ("fonction", "ordre"), [(sp.exp(Z), 0), (Z - ALPHA, 1), (sp.sin(Z) ** 2, 2)]
)
def test_l_ordre_du_zero(fonction: sp.Expr, ordre: int) -> None:
    """Page 27 : `h` admet un zero d'ordre `m >= 0` en `a`."""
    point = ALPHA if fonction.has(ALPHA) else 0
    assert ordre_du_zero(fonction, Z, point) == ordre


def test_le_cas_m_egal_un_de_la_page_29() -> None:
    """Pages 28 et 29, a) : `g(a) / h'(a)`, `g = -i / (z - beta)` et `h = z - alpha`."""
    obtenu = residu_d_un_quotient(-sp.I / (Z - BETA), Z - ALPHA, Z, ALPHA)
    assert sp.simplify(obtenu + sp.I / (ALPHA - BETA)) == 0


@pytest.mark.parametrize(
    ("g", "h"),
    [(sp.exp(Z), sp.sin(Z) ** 2), (sp.cos(Z), Z**2 * (1 + Z))],
)
def test_le_cas_m_egal_deux_de_la_page_28(g: sp.Expr, h: sp.Expr) -> None:
    """Page 28, b) : `(2 g' h'' - (2/3) g h''') / (h'')^2`, les derivees en `a`."""

    def en_zero(fonction: sp.Expr, ordre: int) -> sp.Expr:
        return sp.diff(fonction, Z, ordre).subs(Z, 0)

    formule = (
        2 * en_zero(g, 1) * en_zero(h, 2)
        - sp.Rational(2, 3) * en_zero(g, 0) * en_zero(h, 3)
    ) / en_zero(h, 2) ** 2
    assert residu_d_un_quotient(g, h, Z, 0) == formule


def test_un_zero_triple_par_l_algorithme_et_par_la_methode_3() -> None:
    """Page 28 : « les cas m > 2 se calculent de meme » ; recoupe par la methode 3."""
    g, h = sp.exp(2 * Z), sp.sin(Z) ** 3
    assert residu_d_un_quotient(g, h, Z, 0) == residu_d_un_pole(g / h, Z, 0, ordre=3)


def test_un_denominateur_sans_zero_donne_un_residu_nul() -> None:
    """Page 27 : « le cas m = 0 [...] on est renvoye au cas 1 »."""
    assert residu_d_un_quotient(sp.exp(Z), 1 + Z, Z, 0) == 0


def test_un_numerateur_nul_au_point_est_refuse() -> None:
    """Page 27 : la methode 4 suppose `g(a)` non nul."""
    with pytest.raises(ValueError, match="s'annule"):
        residu_d_un_quotient(sp.sin(Z), Z**2, Z, 0)
