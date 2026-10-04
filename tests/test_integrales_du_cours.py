"""Mathematique, planches 29 a 42 : primitives et integrales definies.

Oracles (`Mathematique/SLIDES/Slides_01_Algebre_lineaire_et_integrales.pdf`) :
la somme de Riemann de la planche 30 ; les proprietes de la planche 32 ;
les quatorze derivees de la planche 33 ; les exemples des planches 35 a
42 -- `int x/(x^2+1)^2`, `int tan x`, `int x e^x`, `int 1/(x^2-4)`,
`int_0^1 ln x = -1`, `int x sin x`, `int x^3 e^(2x)`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.integration.derivees import (
    DERIVEES_USUELLES,
    derivee_usuelle,
)
from polytech.maths.domain.integration.fractions import FractionRationnelle
from polytech.maths.domain.integration.parties import IntegrationParParties
from polytech.maths.domain.integration.proprietes import integrale_orientee
from polytech.maths.domain.integration.riemann import (
    limite_des_sommes,
    somme_de_riemann,
)
from polytech.maths.domain.integration.singulieres import IntegraleSinguliere
from polytech.maths.domain.integration.substitution import Substitution
from polytech.maths.domain.symboles import ENTIER as N
from polytech.maths.domain.symboles import VARIABLE as X

R = sp.Rational


def _derive_en(primitive: sp.Expr, integrande: sp.Expr) -> bool:
    """Le seul controle d'une primitive : la deriver."""
    return bool(sp.simplify(sp.diff(primitive, X) - integrande) == 0)


def test_une_somme_de_riemann_additionne_les_rectangles() -> None:
    """Planche 30 : `sum f(xi_i) dx_i` ; `x^2` sur `[0, 1]`, quatre rectangles."""
    attendu = R(1, 4) * (0 + R(1, 16) + R(4, 16) + R(9, 16))
    assert somme_de_riemann(X**2, (0, 1), 4) == attendu


def test_la_somme_de_riemann_tend_vers_l_integrale() -> None:
    """Planches 30-31 : la limite des sommes est `F(b) - F(a)` ; ici 1/3."""
    assert (
        sp.simplify(
            somme_de_riemann(X**2, (0, 1), N) - (N - 1) * (2 * N - 1) / (6 * N**2)
        )
        == 0
    )
    assert limite_des_sommes(X**2, (0, 1)) == R(1, 3)
    assert limite_des_sommes(X**2, (0, 1)) == integrale_orientee(X**2, 0, 1)


def test_un_nombre_de_rectangles_nul_est_refuse() -> None:
    with pytest.raises(ValueError, match="au moins un rectangle"):
        somme_de_riemann(X, (0, 1), 0)


def test_l_integrale_est_lineaire() -> None:
    """Planche 32 : `int (c f + d g) = c int f + d int g`."""
    f, g = sp.sin(X), X**2
    gauche = integrale_orientee(3 * f - 2 * g, 0, 1)
    droite = 3 * integrale_orientee(f, 0, 1) - 2 * integrale_orientee(g, 0, 1)
    assert sp.simplify(gauche - droite) == 0


def test_l_integrale_se_decoupe_en_un_point_intermediaire() -> None:
    """Planche 32 : `int_a^b = int_a^c + int_c^b`, meme si `c` est hors de `[a, b]`."""
    f = X**3 - X
    for c in (R(1, 2), 5):
        somme = integrale_orientee(f, 0, c) + integrale_orientee(f, c, 2)
        assert somme == integrale_orientee(f, 0, 2) == 2


def test_inverser_les_bornes_change_le_signe() -> None:
    """Planche 32 : `int_a^b = -int_b^a` et `int_a^a = 0`."""
    assert integrale_orientee(X**2, 1, 0) == -integrale_orientee(X**2, 0, 1) == R(-1, 3)
    assert integrale_orientee(X**2, 3, 3) == 0


def test_la_table_des_derivees_porte_les_quatorze_formules_de_la_planche_33() -> None:
    attendues = {
        sp.sin(X): sp.cos(X),
        sp.cos(X): -sp.sin(X),
        sp.tan(X): 1 / sp.cos(X) ** 2,
        sp.exp(X): sp.exp(X),
        sp.sinh(X): sp.cosh(X),
        sp.cosh(X): sp.sinh(X),
        sp.tanh(X): 1 / sp.cosh(X) ** 2,
        sp.asin(X): 1 / sp.sqrt(1 - X**2),
        sp.acos(X): -1 / sp.sqrt(1 - X**2),
        sp.atan(X): 1 / (1 + X**2),
        sp.log(X): 1 / X,
        sp.asinh(X): 1 / sp.sqrt(X**2 + 1),
        sp.acosh(X): 1 / sp.sqrt(X**2 - 1),
        sp.atanh(X): 1 / (1 - X**2),
    }
    assert attendues == DERIVEES_USUELLES


@pytest.mark.parametrize("fonction", list(DERIVEES_USUELLES))
def test_chaque_formule_de_la_table_se_retrouve_en_derivant(fonction: sp.Expr) -> None:
    """La table est une lecture de la planche ; le calcul la controle.

    L'egalite est verifiee en deux points du domaine de la fonction :
    `argcosh` n'est definie que pour `x > 1`, les fonctions `arc` de la
    trigonometrie circulaire que pour `|x| < 1`.
    """
    points = (2, 3) if fonction == sp.acosh(X) else (R(1, 3), R(1, 2))
    ecart = sp.diff(fonction, X) - derivee_usuelle(fonction)
    assert all(sp.simplify(ecart.subs(X, point)) == 0 for point in points)


def test_une_fonction_hors_de_la_table_est_nommee() -> None:
    with pytest.raises(ValueError, match="n'est pas dans la table"):
        derivee_usuelle(sp.sec(X))


def test_le_changement_de_variable_de_la_planche_35() -> None:
    """`u = x^2 + 1` : `1/2 int du/u^2 = -1/(2(x^2 + 1))`."""
    substitution = Substitution(X / (X**2 + 1) ** 2, X**2 + 1)
    u = substitution.nouvelle_variable()
    assert substitution.differentielle() == 2 * X
    assert substitution.integrale_transformee() == 1 / (2 * u**2)
    assert sp.simplify(substitution.resultat() + 1 / (2 * (X**2 + 1))) == 0


def test_le_changement_de_variable_de_la_planche_36() -> None:
    """`int tan x = -ln|cos x| + c`, en posant `u = cos x`."""
    substitution = Substitution(sp.tan(X), sp.cos(X))
    assert substitution.differentielle() == -sp.sin(X)
    assert _derive_en(substitution.resultat(), sp.tan(X))
    assert _derive_en(-sp.log(sp.cos(X)), sp.tan(X))


def test_l_integration_par_parties_de_la_planche_38() -> None:
    """`u = x`, `dv = e^x dx` : `x e^x - int e^x dx = (x - 1) e^x`."""
    parties = IntegrationParParties(X, sp.exp(X))
    assert (parties.derivee_de_u(), parties.v()) == (1, sp.exp(X))
    assert parties.etape() == X * sp.exp(X) - sp.Integral(sp.exp(X), X)
    assert sp.simplify(parties.resultat() - (X - 1) * sp.exp(X)) == 0


def test_l_exercice_resolu_de_la_planche_41() -> None:
    """`int x sin x = -x cos x + sin x + C`."""
    parties = IntegrationParParties(X, sp.sin(X))
    assert parties.v() == -sp.cos(X)
    assert sp.simplify(parties.resultat() - (-X * sp.cos(X) + sp.sin(X))) == 0


def test_l_exercice_resolu_de_la_planche_42_demande_trois_integrations() -> None:
    """`int x^3 e^(2x)` : la premiere etape, puis le resultat final."""
    parties = IntegrationParParties(X**3, sp.exp(2 * X))
    etape = X**3 * sp.exp(2 * X) / 2 - sp.Integral(3 * X**2 * sp.exp(2 * X) / 2, X)
    assert sp.simplify(parties.etape() - etape) == 0
    attendu = sp.exp(2 * X) * (X**3 / 2 - 3 * X**2 / 4 + 3 * X / 4 - R(3, 8))
    assert sp.simplify(parties.resultat() - attendu) == 0
    assert _derive_en(attendu, X**3 * sp.exp(2 * X))


def test_la_decomposition_en_fractions_simples_de_la_planche_39() -> None:
    """`1/(x^2 - 4) = 1/(4(x - 2)) - 1/(4(x + 2))`, `A = 1/4`, `B = -1/4`."""
    fraction = FractionRationnelle(sp.Integer(1), X**2 - 4)
    termes = sorted(fraction.termes(), key=lambda terme: terme.subs(X, 3))
    attendus = (-1 / (4 * (X + 2)), 1 / (4 * (X - 2)))
    assert len(termes) == len(attendus)
    assert all(sp.simplify(t - a) == 0 for t, a in zip(termes, attendus, strict=True))
    assert _derive_en(fraction.primitive(), 1 / (X**2 - 4))
    assert _derive_en((sp.log(X - 2) - sp.log(X + 2)) / 4, 1 / (X**2 - 4))


def test_la_derniere_ligne_de_la_planche_39_a_perdu_son_logarithme() -> None:
    """La planche ecrit `1/4 |(x-2)/(x+2)| + C` ; c'est `1/4 ln|(x-2)/(x+2)| + C`.

    L'expression imprimee n'est pas une primitive de `1/(x^2 - 4)` : sa
    derivee, en `x = 3`, vaut `1/25` et non `1/5`.
    """
    imprimee = (X - 2) / (X + 2) / 4
    corrigee = sp.log((X - 2) / (X + 2)) / 4
    assert sp.diff(imprimee, X).subs(X, 3) == R(1, 25)
    assert (1 / (X**2 - 4)).subs(X, 3) == R(1, 5)
    assert not _derive_en(imprimee, 1 / (X**2 - 4))
    assert _derive_en(corrigee, 1 / (X**2 - 4))


def test_l_integrale_impropre_de_la_planche_40() -> None:
    """`int_0^1 ln x dx = lim [x ln x - x]_delta^1 = -1`."""
    integrale = IntegraleSinguliere(sp.log(X), singularite=0, autre_borne=1)
    delta = integrale.borne_mobile()
    assert (
        sp.simplify(
            integrale.primitive_bornee() - (-1 - (delta * sp.log(delta) - delta))
        )
        == 0
    )
    assert integrale.converge()
    assert integrale.valeur() == -1


def test_une_integrale_singuliere_divergente_est_dite_divergente() -> None:
    """`int_0^1 dx/x` : `ln delta` n'a pas de limite finie."""
    integrale = IntegraleSinguliere(1 / X, singularite=0, autre_borne=1)
    assert not integrale.converge()
    with pytest.raises(ValueError, match="l'integrale diverge"):
        integrale.valeur()


def test_une_singularite_a_la_borne_haute_s_approche_par_en_dessous() -> None:
    """`int_0^1 dx / sqrt(1 - x) = 2` : la borne mobile tend vers 1 par la gauche."""
    integrale = IntegraleSinguliere(1 / sp.sqrt(1 - X), singularite=1, autre_borne=0)
    assert integrale.valeur() == 2


def test_la_somme_de_riemann_sur_un_intervalle_qui_ne_part_pas_de_zero() -> None:
    """`x` sur `[1, 3]`, deux rectangles de largeur 1 : `f(1) + f(2) = 3`."""
    assert somme_de_riemann(X, (1, 3), 2) == 3


def test_la_borne_mobile_approche_la_singularite_de_l_interieur() -> None:
    """`int_0^1 e^(-1/x) / x^2 dx = 1/e` ; par la gauche de 0, elle divergerait."""
    integrale = IntegraleSinguliere(sp.exp(-1 / X) / X**2, singularite=0, autre_borne=1)
    assert integrale.valeur() == sp.exp(-1)
