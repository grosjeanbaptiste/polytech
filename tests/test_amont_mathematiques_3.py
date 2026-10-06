"""Mathematiques III : les exemples chiffres du cours, refaits par la bibliotheque.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que les symboles auxquels `cursus/notions/MathematiquesIngenieur3.toml`
rattache un titre portent bien ce que la planche enonce -- en recopiant ses
exemples, planche citee.

Sources : `MathematiquesIngenieur3/SLIDES/`, sections 1, 2, 10 et 11.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.complexes import racines_niemes
from polytech.maths.domain.algebre.complexes_exacts import par_moivre
from polytech.maths.domain.laplace.coefficients_variables import du_produit_par_t
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.series.suites_de_fonctions import tente
from polytech.maths.domain.symboles import IMAGE, TEMPS, VARIABLE

OMEGA = sp.Symbol("omega", positive=True)
A = sp.Symbol("a", positive=True)


def test_la_formule_de_moivre_de_la_planche_8() -> None:
    """Section 1, planche 8 : `(cos t + i sin t)^n = cos(nt) + i sin(nt)`."""
    z = sp.cos(sp.pi / 6) + sp.I * sp.sin(sp.pi / 6)
    assert sp.simplify(par_moivre(z, 3) - sp.I) == 0


@pytest.mark.parametrize("ordre", [2, 3, 5])
def test_chaque_racine_nieme_resout_l_equation_de_la_planche_10(ordre: int) -> None:
    """Section 2, (2.4) : les racines n-iemes de `z` verifient `zeta^n = z`."""
    z = complex(-8, 3)
    racines = racines_niemes(z, ordre)
    assert len(racines) == ordre
    for racine in racines:
        assert racine**ordre == pytest.approx(z, rel=1e-12)


@pytest.mark.parametrize("rang", [2, 5, 10])
def test_la_tente_de_la_section_10_garde_une_aire_d_un(rang: int) -> None:
    """Partie 2, planche 16, « limite des integrales != integrale de la limite ».

    `s_k` vaut `k2 x` sur `]0, 1/k]`, `2k - k2 x` sur `]1/k, 2/k]`, zero
    ensuite : son integrale sur `[0, 2]` vaut 1 pour tout `k`, son maximum
    vaut `k`, et elle est deja nulle en tout point fixe a droite de `2/k`.
    """
    s_k = tente(rang)
    assert sp.integrate(s_k, (VARIABLE, 0, 2)) == 1
    assert s_k.subs(VARIABLE, sp.Rational(1, rang)) == rang
    assert s_k.subs(VARIABLE, 1) == 0


def test_la_transformee_de_t_carre_de_la_planche_6() -> None:
    """Section 11.II, planche 7 : `L(t2 nu(t))(z) = 2/z3`."""
    assert Transformee.de(TEMPS**2).image == 2 / IMAGE**3


def test_la_meme_transformee_par_la_derivee_de_l_image() -> None:
    """Planche 11, version 1 : `V'' = L(t2 nu)`, avec `V = 1/z` et `F' = -L(t f)`."""
    assert sp.simplify(du_produit_par_t(1 / IMAGE, 2) - 2 / IMAGE**3) == 0


@pytest.mark.parametrize(
    ("original", "image"),
    [
        (sp.Integer(1), 1 / IMAGE),
        (sp.exp(-A * TEMPS), 1 / (IMAGE + A)),
        (TEMPS**3, 6 / IMAGE**4),
        (sp.sin(OMEGA * TEMPS), OMEGA / (IMAGE**2 + OMEGA**2)),
        (sp.cos(OMEGA * TEMPS), IMAGE / (IMAGE**2 + OMEGA**2)),
    ],
)
def test_la_table_des_transformees_usuelles_de_la_planche_12(
    original: sp.Expr, image: sp.Expr
) -> None:
    """Section 11.II, planche 13 : `nu`, `e^(-at)`, `t^n`, `sin`, `cos`."""
    assert sp.simplify(Transformee.de(original).image - image) == 0
