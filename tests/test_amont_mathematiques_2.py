"""Mathematiques II : les exemples chiffres du cours, refaits par la bibliotheque.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que les symboles auxquels
`cursus/notions/CalculDifferentielIntegral2.toml` rattache un titre portent
ce que la planche enonce -- en recopiant ses exemples, page citee.

Source : `CalculDifferentielIntegral2/SLIDES/Slides_08_Green_Stokes_Ostrogradsky.pdf`.
Les pages sont celles du PDF ; la planche imprimee est entre parentheses.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.analyse_vectorielle import divergence, rotationnel

X, Y, Z = sp.symbols("x y z", real=True)
ESPACE = (X, Y, Z)
RAYON, ANGLE = sp.symbols("r theta", nonnegative=True)
CHAMP_DE_STOKES = (-(Y**3), X**3, -(Z**3))
"""Pages 56 a 58 (53 a 55) : l'exercice resolu du theoreme de Stokes."""


def test_le_rotationnel_est_celui_de_la_page_44() -> None:
    """Page 44 (41) : `(dR/dy - dQ/dz, dP/dz - dR/dx, dQ/dx - dP/dy)`."""
    p, q, r = (sp.Function(nom)(*ESPACE) for nom in "PQR")
    attendu = (
        sp.diff(r, Y) - sp.diff(q, Z),
        sp.diff(p, Z) - sp.diff(r, X),
        sp.diff(q, X) - sp.diff(p, Y),
    )
    assert rotationnel((p, q, r), ESPACE) == attendu


def test_dans_le_plan_stokes_rend_l_integrande_de_green() -> None:
    """Page 47 (44) : `rot F | (0, 0, 1) = dQ/dx - dP/dy`."""
    p, q = (sp.Function(nom)(X, Y) for nom in "PQ")
    verticale = rotationnel((p, q, sp.Integer(0)), ESPACE)[2]
    assert verticale == sp.diff(q, X) - sp.diff(p, Y)


def test_le_rotationnel_de_l_exercice_resolu_de_stokes() -> None:
    """Page 57 (54) : `rot(-y3, x3, -z3) = 3(x2 + y2) u_z`."""
    assert rotationnel(CHAMP_DE_STOKES, ESPACE) == (0, 0, 3 * (X**2 + Y**2))


def test_la_circulation_de_l_exercice_resolu_de_stokes_vaut_trois_pi_sur_deux() -> None:
    """Pages 57 et 58 (54 et 55) : la planche imprime `pi / 2`, et c'est un ecart.

    Elle ecrit la normale unitaire `(1/3)(1, 1, 1)` ; elle vaut
    `(1, 1, 1) / racine(3)`, et l'element d'aire de `z = 1 - x - y` est
    `racine(3) dx dy`. Le flux du rotationnel est donc celui de
    `rot F | (1, 1, 1) = 3(x2 + y2)` sur le disque unite : `3 pi / 2`.
    Second chemin, sans Stokes : la circulation le long de
    `(cos t, sin t, 1 - cos t - sin t)`.
    """
    tourbillon = rotationnel(CHAMP_DE_STOKES, ESPACE)
    normale = (1, 1, 1)
    integrande = sum(a * b for a, b in zip(tourbillon, normale, strict=True))
    polaire = sp.simplify(
        integrande.subs({X: RAYON * sp.cos(ANGLE), Y: RAYON * sp.sin(ANGLE)})
    )
    par_stokes = sp.integrate(polaire * RAYON, (RAYON, 0, 1), (ANGLE, 0, 2 * sp.pi))
    chemin = (sp.cos(ANGLE), sp.sin(ANGLE), 1 - sp.cos(ANGLE) - sp.sin(ANGLE))
    sur_le_chemin = dict(zip(ESPACE, chemin, strict=True))
    travail = sum(
        composante.subs(sur_le_chemin) * sp.diff(coordonnee, ANGLE)
        for composante, coordonnee in zip(CHAMP_DE_STOKES, chemin, strict=True)
    )
    directe = sp.integrate(travail, (ANGLE, 0, 2 * sp.pi))
    assert par_stokes == directe == 3 * sp.pi / 2


@pytest.mark.parametrize(
    ("champ", "attendue"),
    [
        ((X * Y**2, X**2 * Y, Y), X**2 + Y**2),
        ((X, sp.Integer(1), sp.Integer(1)), sp.Integer(1)),
        ((X * Y, Y**2 + sp.exp(X * Z**2), sp.sin(X * Y)), 3 * Y),
    ],
    ids=["ex 1, page 73", "ex 2, page 75", "ex 3, page 78"],
)
def test_la_divergence_des_trois_exercices_resolus(
    champ: tuple[sp.Expr, sp.Expr, sp.Expr], attendue: sp.Expr
) -> None:
    """Pages 73, 75 et 78 (69, 71 et 73) : la divergence que la solution ecrit."""
    assert sp.simplify(divergence(champ, ESPACE) - attendue) == 0


def test_le_flux_de_l_exercice_resolu_1_vaut_pi() -> None:
    """Page 73 (69) : `int div F` sur le cylindre de rayon 1, `-1 <= z <= 1`."""
    source = divergence((X * Y**2, X**2 * Y, Y), ESPACE)
    polaire = sp.simplify(
        source.subs({X: RAYON * sp.cos(ANGLE), Y: RAYON * sp.sin(ANGLE)})
    )
    flux = sp.integrate(
        polaire * RAYON, (RAYON, 0, 1), (Z, -1, 1), (ANGLE, 0, 2 * sp.pi)
    )
    assert flux == sp.pi


def test_le_flux_de_l_exercice_resolu_3_vaut_184_sur_35() -> None:
    """Page 78 (73) : `0 <= y <= 2 - z`, `0 <= z <= 1 - x2`, `-1 <= x <= 1`."""
    source = divergence((X * Y, Y**2 + sp.exp(X * Z**2), sp.sin(X * Y)), ESPACE)
    flux = sp.integrate(source, (Y, 0, 2 - Z), (Z, 0, 1 - X**2), (X, -1, 1))
    assert flux == sp.Rational(184, 35)
