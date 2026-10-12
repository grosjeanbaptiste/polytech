"""Les notes de seance de mathematique : chaque ligne du manuscrit, recalculee.

Sources, dans `Mathematique/EXS/` :
`Exs_01_Integration_matrices_determinants_Notes_de_seance.pdf` et
`Exs_02_Equations_differentielles_Notes_de_seance.pdf`,
deux scans manuscrits de six pages, transcrits a la main dans
`transcriptions/Mathematique/EXS/`.

Une transcription est une lecture, et une lecture se trompe en silence. Aucun
test ne recopie donc une ligne pour la comparer a elle-meme : une primitive
est derivee, une solution est reportee dans son equation, une decomposition
est recomposee. Les reponses imprimees du recueil servent de second temoin
la ou il en donne.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sympy as sp

from polytech.maths.domain.integration.fractions import FractionRationnelle
from polytech.maths.domain.symboles import VARIABLE

RACINE = Path(__file__).resolve().parent.parent
NOTES = RACINE / "transcriptions" / "Mathematique" / "EXS"
T, X, U, Z = sp.symbols("t x u z", real=True)
A, B = sp.symbols("a b", positive=True)
C, K = sp.symbols("C K", real=True)


def _est_une_primitive(
    primitive: sp.Expr, integrande: sp.Expr, variable: sp.Symbol
) -> bool:
    return bool(sp.simplify(sp.diff(primitive, variable) - integrande) == 0)


# --- Chapitre 1 : integration -----------------------------------------------


def test_cos_5_devient_un_polynome_en_sinus() -> None:
    """Exercice 2 : `cos^5 = (1 - sin^2)^2 cos`, puis `u = sin x`."""
    assert sp.simplify(sp.cos(X) ** 5 - (1 - sp.sin(X) ** 2) ** 2 * sp.cos(X)) == 0
    primitive = sp.integrate((1 - U**2) ** 2, U).subs(U, sp.sin(X))
    assert _est_une_primitive(primitive, sp.cos(X) ** 5, X)


def test_la_linearisation_de_cos_4() -> None:
    """Les trois lignes de la page 1, puis la primitive."""
    lineaire = (1 + 2 * sp.cos(2 * T) + sp.Rational(1, 2) + sp.cos(4 * T) / 2) / 4
    assert sp.simplify(sp.cos(T) ** 4 - lineaire) == 0
    primitive = (T + sp.sin(2 * T) + T / 2 + sp.sin(4 * T) / 8) / 4
    assert _est_une_primitive(primitive, sp.cos(T) ** 4, T)


def test_la_primitive_du_logarithme() -> None:
    """Exercice 4 : `x ln x - x`, et la reponse imprimee du recueil."""
    positif = sp.Symbol("x", positive=True)
    assert _est_une_primitive(
        positif * sp.log(positif) - positif, sp.log(positif), positif
    )


def test_les_formules_de_l_arc_moitie() -> None:
    """Exercice 6 : les deux « formules magiques »."""
    moitie = sp.tan(X / 2)
    assert sp.simplify(sp.sin(X) - 2 * moitie / (1 + moitie**2)) == 0
    assert sp.simplify(sp.cos(X) - (1 - moitie**2) / (1 + moitie**2)) == 0


def test_l_inverse_du_sinus_s_integre_en_logarithme_de_la_tangente() -> None:
    """`du = (1 + tan^2(x/2)) / 2 dx` ramene l'integrale a celle de `1 / u`."""
    moitie = sp.tan(X / 2)
    assert sp.simplify(sp.diff(moitie, X) - (1 + moitie**2) / 2) == 0
    entre_zero_et_pi = sp.Symbol("x", positive=True)
    primitive = sp.log(sp.tan(entre_zero_et_pi / 2))
    assert _est_une_primitive(primitive, 1 / sp.sin(entre_zero_et_pi), entre_zero_et_pi)


def test_l_exponentielle_fois_le_cosinus() -> None:
    """Exercice 9 : le resultat du manuscrit, et celui du recueil."""
    crochet = (
        sp.exp(A * X) * sp.sin(B * X) / B + A * sp.exp(A * X) * sp.cos(B * X) / B**2
    )
    manuscrit = crochet / (1 + A**2 / B**2)
    assert _est_une_primitive(manuscrit, sp.exp(A * X) * sp.cos(B * X), X)
    recueil = sp.exp(A * X) * (A * sp.cos(B * X) + B * sp.sin(B * X)) / (A**2 + B**2)
    assert sp.simplify(manuscrit - recueil) == 0


def test_la_seconde_integration_par_parties_change_bien_de_signe() -> None:
    """Page 4 : `(a/b) (-a/b) I` ; un signe perdu donnerait `1 - a^2 / b^2`."""
    seconde = sp.integrate(sp.exp(A * X) * sp.sin(B * X), X)
    par_parties = -sp.exp(A * X) * sp.cos(B * X) / B + (A / B) * sp.integrate(
        sp.exp(A * X) * sp.cos(B * X), X
    )
    assert sp.simplify(seconde - par_parties) == 0


def test_le_conjugue_hyperbolique() -> None:
    """Exercice 14 : le denominateur vaut un, et il reste deux integrales."""
    assert sp.simplify(sp.cosh(X) ** 2 - sp.sinh(X) ** 2) == 1
    integrande = sp.cosh(X) / (sp.cosh(X) + sp.sinh(X))
    separe = sp.cosh(X) ** 2 - sp.cosh(X) * sp.sinh(X)
    assert sp.simplify(integrande - separe) == 0
    recueil = X / 2 + sp.sinh(2 * X) / 4 - sp.cosh(2 * X) / 4
    assert _est_une_primitive(recueil, integrande, X)


def test_l_integrale_definie_de_cos_4() -> None:
    """Serie 1.2 : la primitive regroupee en `3t / 2`, puis `3 pi / 16`."""
    regroupee = (3 * T / 2 + sp.sin(2 * T) + sp.sin(4 * T) / 8) / 4
    assert _est_une_primitive(regroupee, sp.cos(T) ** 4, T)
    valeur = regroupee.subs(T, sp.pi / 2) - regroupee.subs(T, 0)
    assert valeur == 3 * sp.pi / 16
    assert sp.integrate(sp.cos(T) ** 4, (T, 0, sp.pi / 2)) == 3 * sp.pi / 16


# --- Chapitre 2 : equations differentielles ---------------------------------


@pytest.mark.parametrize("signe", [1, -1])
def test_exercice_1_la_solution_verifie_l_equation(signe: int) -> None:
    y = signe * sp.sqrt(1 + K * sp.exp(X**2))
    assert sp.simplify(y * sp.diff(y, X) - X * y**2 + X) == 0


def test_exercice_1_la_constante_nulle_est_une_solution() -> None:
    """La correction rouge : `K = 0` donne `y = 1` et `y = -1`."""
    for constante in (sp.Integer(1), sp.Integer(-1)):
        assert constante * sp.diff(constante, X) - X * constante**2 + X == 0


def test_exercice_2_le_changement_de_fonction() -> None:
    """`z = y / x` : `x z' = (-z^3 - 2z) / (1 + z^2)`."""
    assert sp.simplify(-Z / (1 + Z**2) - Z - (-(Z**3) - 2 * Z) / (1 + Z**2)) == 0


def test_exercice_2_la_decomposition_en_fractions_simples() -> None:
    """`A = 1/2`, `B = 1/2`, `C = 0`, par la bibliotheque et par recomposition."""
    v = VARIABLE
    termes = FractionRationnelle(1 + v**2, v**3 + 2 * v).termes()
    attendus = (1 / (2 * v), v / (2 * (v**2 + 2)))
    assert len(termes) == len(attendus)
    assert sp.simplify(sum(termes) - sum(attendus)) == 0
    assert any(sp.simplify(terme - attendus[0]) == 0 for terme in termes)
    primitive = sp.log(sp.Abs(Z)) / 2 + sp.log(Z**2 + 2) / 4
    positif = sp.Symbol("z", positive=True)
    derivee = sp.diff(primitive.subs(Z, positif), positif)
    assert sp.simplify(derivee - (1 + positif**2) / (positif**3 + 2 * positif)) == 0


def test_exercice_2_la_solution_implicite_verifie_l_equation() -> None:
    """`y sqrt(y^2 + 2 x^2) = K` : la pente de ses courbes est celle de l'equation."""
    y = sp.Symbol("y", positive=True)
    integrale = y * sp.sqrt(y**2 + 2 * X**2)
    pente = -sp.diff(integrale, X) / sp.diff(integrale, y)
    assert sp.simplify(pente + X * y / (X**2 + y**2)) == 0


def test_exercice_5_la_formule_du_premier_ordre() -> None:
    """La solution du manuscrit, et la forme `sin x - 1 + K e^(-sin x)` du recueil."""
    assert _est_une_primitive(U * sp.exp(U) - sp.exp(U), U * sp.exp(U), U)
    y = sp.exp(-sp.sin(X)) * (C + sp.sin(X) * sp.exp(sp.sin(X)) - sp.exp(sp.sin(X)))
    assert sp.simplify(sp.diff(y, X) + y * sp.cos(X) - sp.cos(X) * sp.sin(X)) == 0
    assert sp.simplify(y - (sp.sin(X) - 1 + C * sp.exp(-sp.sin(X)))) == 0


def test_exercice_10_la_solution_verifie_l_equation() -> None:
    """`x y^2 (x y' + y) = a`, avec `z = x y`."""
    positif = sp.Symbol("x", positive=True)
    y = sp.cbrt(3) / positif * sp.cbrt(A * positif**2 / 2 + C)
    residu = positif * y**2 * (positif * sp.diff(y, positif) + y) - A
    assert sp.simplify(residu) == 0


# --- Les transcriptions elles-memes -----------------------------------------


@pytest.mark.parametrize(
    ("fichier", "resultats"),
    [
        (
            "Exs_01_Integration_matrices_determinants_Notes_de_seance.md",
            (r"\frac{3\pi}{16}", r"x\ln(x) - x + C", r"\ln|u| + C"),
        ),
        (
            "Exs_02_Equations_differentielles_Notes_de_seance.md",
            (r"y\,\sqrt{y^2 + 2x^2} = K", r"\pm\sqrt{1 + e^{x^2}K}", r"\sqrt[3]{3}"),
        ),
    ],
)
def test_la_transcription_porte_les_resultats_verifies(
    fichier: str, resultats: tuple[str, ...]
) -> None:
    """Ce que ce fichier recalcule est bien ce que la transcription ecrit."""
    texte = (NOTES / fichier).read_text(encoding="utf-8")
    assert texte.startswith("<!--\nTranscription visuelle d'un scan manuscrit")
    assert [r for r in resultats if r not in texte] == []


# --- Les annotations du fascicule 2 -----------------------------------------
#
# `Mathematique/SLIDES/Slides_02_Equations_differentielles_annote.pdf` : les
# derivations ecrites a la main sur les planches, recalculees une a une.

GAMMA, ALPHA = sp.symbols("gamma alpha", real=True)
Y = sp.Symbol("y", real=True)


def test_planche_17_le_seau_troue_par_separation() -> None:
    """`2 sqrt(h) = gamma t - K`, soit `h = gamma^2 (t - alpha)^2 / 4`."""
    hauteur = GAMMA**2 * (T - ALPHA) ** 2 / 4
    residu = sp.diff(hauteur, T) ** 2 - GAMMA**2 * hauteur
    assert sp.simplify(residu) == 0
    assert sp.simplify(2 * sp.sqrt(hauteur) ** 2 - 2 * hauteur) == 0
    constante = sp.solve(sp.Eq(GAMMA * T - K, GAMMA * (T - ALPHA)), K)
    assert constante == [ALPHA * GAMMA]


def test_planche_21_le_potentiel_de_l_equation_exacte() -> None:
    """`x^3 / 3 + x y + C(y)`, puis `C'(y) = -2 y`."""
    partielle = sp.integrate(X**2 + Y, X)
    assert sp.expand(partielle - (X**3 / 3 + Y * X)) == 0
    potentiel = X**3 / 3 + X * Y - Y**2
    assert sp.diff(potentiel, X) == X**2 + Y
    assert sp.diff(potentiel, Y) == X - 2 * Y


def test_planche_22_la_condition_necessaire_d_exactitude() -> None:
    """`dB/dx = dA/dy`, verifiee sur l'exemple de la planche 21."""
    assert sp.diff(X - 2 * Y, X) == sp.diff(X**2 + Y, Y)


def test_planche_24_l_exercice_laisse_en_suspens() -> None:
    """`dy / (3 - 2y) = dx` : la solution verifie `y' + 2y = 3`."""
    y = sp.Rational(3, 2) + C * sp.exp(-2 * X)
    assert sp.simplify(sp.diff(y, X) + 2 * y - 3) == 0
    assert _est_une_primitive(-sp.log(3 - 2 * U) / 2, 1 / (3 - 2 * U), U)


def test_planche_27_l_isocline_de_pente_trois_demis() -> None:
    assert sp.solve(sp.Eq(X - Y**2, sp.Rational(3, 2)), X) == [Y**2 + sp.Rational(3, 2)]


def test_planche_31_la_solution_qui_explose() -> None:
    """`-1 / y = t + C` : `y = -1 / (t + C)` verifie `y' = y^2`."""
    y = -1 / (T + C)
    assert sp.simplify(sp.diff(y, T) - y**2) == 0
    assert sp.simplify(-1 / y - (T + C)) == 0


def test_planche_36_x_y_carre_est_homogene_de_degre_trois() -> None:
    echelle = sp.Symbol("lambda", positive=True)
    f = X * Y**2
    rapport = f.subs({X: echelle * X, Y: echelle * Y}, simultaneous=True) / f
    assert sp.simplify(rapport) == echelle**3


def test_planche_37_l_equation_homogene_de_degre_zero() -> None:
    """Les trois lignes du changement `z = y / x`."""
    echelle = sp.Symbol("lambda", positive=True)
    f = 2 * X * Y / (X**2 - Y**2)
    assert (
        sp.simplify(f.subs({X: echelle * X, Y: echelle * Y}, simultaneous=True) - f)
        == 0
    )
    assert sp.simplify(f.subs(Y, Z * X) - 2 * Z / (1 - Z**2)) == 0
    assert sp.simplify(2 * Z / (1 - Z**2) - Z - (Z + Z**3) / (1 - Z**2)) == 0


def test_planche_39_la_formule_du_premier_ordre_se_verifie() -> None:
    """`y = e^(-int P) [C + int Q e^(int P)]`, sur `P = 2`, `Q = 3`."""
    primitive = 2 * X
    y = sp.exp(-primitive) * (C + sp.integrate(3 * sp.exp(primitive), X))
    assert sp.simplify(sp.diff(y, X) + 2 * y - 3) == 0


def test_planche_41_le_signe_du_second_membre() -> None:
    """La solution imprimee verifie `y' - k y = -r` ; le manuscrit ecrit `r`."""
    k, r, coefficient = sp.symbols("k r lambda", positive=True)
    y = (r + coefficient * sp.exp(k * T)) / k
    assert sp.simplify(sp.diff(y, T) - k * y + r) == 0
    assert sp.simplify(sp.diff(y, T) - k * y - r) != 0
    assert sp.solve(sp.Eq(k * Y - r, 0), Y) == [r / k]


def test_planche_46_le_prix_d_equilibre() -> None:
    k, a, b, g, d = sp.symbols("k alpha beta gamma delta", positive=True)
    prix = sp.Symbol("P")
    equilibre = sp.solve(sp.Eq(k * (a + g) - k * (b + d) * prix, 0), prix)
    assert equilibre == [(a + g) / (b + d)]


def test_la_transcription_du_fascicule_annote_dit_ce_qu_elle_est() -> None:
    chemin = NOTES.parent / "SLIDES" / "Slides_02_Equations_differentielles_annote.md"
    texte = chemin.read_text(encoding="utf-8")
    assert texte.startswith("<!--\nTranscription visuelle d'un scan manuscrit")
    assert r"y' - k\,y = r" in texte
    assert "l'équation imprimée donne" in texte
