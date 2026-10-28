"""Introduction aux sciences de l'ingenieur : dimensions, vecteurs, droites et plans.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que les symboles auxquels
`cursus/notions/IntroductionSciencesIngenieur.toml` rattache un titre portent
ce que le syllabus enonce -- en recopiant ses exemples, page citee.

Source : `IntroductionSciencesIngenieur/SYLLABUS/`
`Syllabus_Introduction_aux_sciences_de_l_ingenieur_2025_2026.pdf`, chapitres
B, M et P, et annexe 5. Les pages sont celles du PDF ; la page imprimee est
entre parentheses.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.fluides.domain.analyse_dimensionnelle import nombre_de_groupes
from polytech.fluides.domain.variables_repetees import dimensions_du_produit
from polytech.maths.domain.algebre import complexes_exacts, metrique
from polytech.maths.domain.algebre.varietes_affines import VarieteAffine
from polytech.maths.domain.series.mac_laurin import coefficients_de_mac_laurin
from polytech.maths.domain.series.taylor import (
    coefficients_de_taylor,
    polynome_de_taylor,
)
from polytech.maths.domain.symboles import VARIABLE

CHUTE = {
    "t": {"T": 1},
    "h": {"L": 1},
    "m": {"M": 1},
    "g": {"L": 1, "T": -2},
}
"""Page 21 (16) : le temps de chute, la hauteur, la masse et la pesanteur."""


def test_le_groupe_sans_dimension_de_la_chute_d_un_corps() -> None:
    """Pages 21 et 23 (16 et 18) : `t2 g / h` ; quatre variables, un groupe."""
    assert dimensions_du_produit({"t": 2, "h": -1, "m": 0, "g": 1}, CHUTE) == {}
    assert dimensions_du_produit({"t": 1, "h": -1, "m": 0, "g": 1}, CHUTE) != {}
    assert nombre_de_groupes(parametres=4, reduction=3) == 1


def test_la_norme_et_l_angle_dans_une_base_orthonormee() -> None:
    """Page 128 (123) : `v . w = v1 w1 + v2 w2 + v3 w3`, la norme, l'angle."""
    v, w = (1.0, 2.0, 2.0), (2.0, 0.0, 0.0)
    assert metrique.produit_scalaire(v, w) == pytest.approx(2)
    assert metrique.norme(v) == pytest.approx(3)
    cosinus = metrique.produit_scalaire(v, w) / (metrique.norme(v) * metrique.norme(w))
    assert math.acos(cosinus) == pytest.approx(math.acos(1 / 3))
    directeurs = [composante / metrique.norme(v) for composante in v]
    assert sum(c * c for c in directeurs) == pytest.approx(1)


def test_la_projection_sur_un_axe_de_la_page_129() -> None:
    """Page 129 (124) : `v_u = v . 1_u`, quelle que soit la norme de l'axe."""
    v = (3.0, 4.0, 0.0)
    assert metrique.projection_orthogonale(v, ((5.0, 0.0, 0.0),)) == pytest.approx(
        (3, 0, 0)
    )


def test_la_droite_par_deux_points_de_la_page_152() -> None:
    """Page 152 (147) : par `(2, 0, 5)` et `(3, 1, 2)`, de direction `(1, 1, -3)`."""
    droite = VarieteAffine.engendree_par(((2, 0, 5), (3, 1, 2)))
    assert droite.est_une_droite()
    assert droite.contient((3, 1, 2))
    assert droite.contient((4, 2, -1))
    assert not droite.est_un_sous_espace()
    parallele = VarieteAffine((0, 0, 0), ((2, 2, -6),))
    assert droite.est_parallele_a(parallele)
    assert not droite.est_egale_a(parallele)


def test_le_plan_par_l_origine_de_la_page_154() -> None:
    """Page 154 (149) : par l'origine, `(-1, 0, 2)` et `(2, 3, -5)`."""
    plan = VarieteAffine.engendree_par(((0, 0, 0), (-1, 0, 2), (2, 3, -5)))
    assert plan.est_un_hyperplan()
    assert plan.est_un_sous_espace()
    assert plan.contient((1, 3, -3))
    droite = VarieteAffine.engendree_par(((0, 0, 0), (-1, 0, 2)))
    commune = plan.intersection(droite)
    assert commune is not None
    assert commune.est_une_droite()


def test_les_developpements_de_mac_laurin_de_l_annexe_5() -> None:
    """Page 168 (163) : `e^x`, `ln(1 + x)`, `sin x`. Ecart au cours dans (5.1).

    La formule generale ecrit `f'(0)`, `f''(0)`, ... devant les puissances de
    `(x - a)` : les derivees se prennent en `a`. Les exemples, eux, sont en 0.
    """
    x = VARIABLE
    f = sp.factorial
    assert coefficients_de_mac_laurin(sp.exp(x), 5) == tuple(1 / f(n) for n in range(5))
    quart = sp.Rational(1, 4)
    assert coefficients_de_mac_laurin(sp.log(1 + x), 5) == (
        0,
        1,
        -sp.Rational(1, 2),
        sp.Rational(1, 3),
        -quart,
    )
    assert coefficients_de_mac_laurin(sp.sin(x), 4) == (0, 1, 0, -1 / f(3))


COEFFICIENTS_DE_EXP_EN_1 = (sp.E, sp.E, sp.E / 2, sp.E / 6)
"""Annexe 5, (5.1) : `f^(n)(1) / n!` pour `exp`, soit `e / n!` (`exp' = exp`)."""


def test_taylor_en_a_prend_les_derivees_en_a_et_non_en_zero() -> None:
    """Annexe 5, (5.1) : en `a = 1`, les coefficients de `(x - a)^n` sont `f^(n)(1)/n!`.

    Pour `f = ln x`, la bibliotheque donne `0, 1, -1/2, 1/3` ; les `f^(n)(0)/n!`
    n'existent pas (`ln` n'est pas defini en 0) : la formule en 0 ne vaut que
    pour `a = 0`.
    """
    coefficients = coefficients_de_taylor(sp.log(VARIABLE), 1, 4)
    assert coefficients == (0, 1, -sp.Rational(1, 2), sp.Rational(1, 3))
    assert coefficients_de_taylor(sp.exp(VARIABLE), 1, 4) == COEFFICIENTS_DE_EXP_EN_1
    polynome = polynome_de_taylor(sp.exp(VARIABLE), 1, 3)
    en_zero = polynome_de_taylor(sp.exp(VARIABLE), 0, 3)  # ce que f^(n)(0) donnerait
    assert sp.simplify(polynome - en_zero) != 0


def test_la_formule_de_moivre_et_les_racines_niemes() -> None:
    """Pages 146 et 147 (141 et 142) : `z^n = |z|^n e^(i n arg z)` ; `zeta^n = z`."""
    z = 1 + sp.I
    assert sp.expand_complex(complexes_exacts.par_moivre(z, 4)) == -4
    racines = complexes_exacts.racines_niemes_exactes(sp.Integer(-4), 2)
    assert {sp.expand_complex(r) for r in racines} == {2 * sp.I, -2 * sp.I}
