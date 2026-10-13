"""Syllabus d'algebre, I.2.2 a I.2.4 : calculer en notation trigonometrique.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 5 a 8. Le
produit multiplie les modules et ajoute les arguments, l'inverse inverse
le module et oppose l'argument, et le syllabus ne ramene pas l'argument
dans un intervalle : il ecrit `15 pi / 4` et `-53 pi / 12` tels quels.

Les exemples 1, 2 et 4 a 7 de I.2.4 sont recalcules, et chaque resultat
est controle en revenant a la forme `a + bi`.

Le lieu de I.3.1 (page 11) : `|z - 5i| = 3` est le cercle de centre
`(0, 5)` et de rayon 3.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.complexes_exacts import (
    forme_polaire,
    lieu_dans_le_plan_de_gauss,
)
from polytech.maths.domain.algebre.complexes_polaires import (
    cercle_du_lieu,
    forme_algebrique,
    inverse_polaire,
    produit_polaire,
    puissance_polaire,
    quotient_polaire,
)

I = sp.I  # noqa: E741 -- l'unite imaginaire, que le syllabus note i
R = sp.Rational
pi, sqrt = sp.pi, sp.sqrt
PREMIER = (sp.Integer(2), -pi / 6)
"""`racine(3) - i`, exemple 1."""
SECOND = (sqrt(2), 3 * pi / 4)
"""`-1 + i`, exemple 2."""


def _egaux(obtenu: sp.Expr, attendu: sp.Expr) -> bool:
    return bool(sp.simplify(sp.expand_complex(obtenu - attendu)) == 0)


def test_les_exemples_1_et_2_ont_la_forme_polaire_du_syllabus() -> None:
    assert forme_polaire(sqrt(3) - I) == PREMIER
    assert forme_polaire(-1 + I) == SECOND


def test_la_forme_algebrique_revient_de_la_notation_trigonometrique() -> None:
    assert _egaux(forme_algebrique(*PREMIER), sqrt(3) - I)
    assert _egaux(forme_algebrique(*SECOND), -1 + I)


def test_l_exemple_4_le_produit_ajoute_les_arguments() -> None:
    produit = produit_polaire(PREMIER, SECOND)
    assert produit == (2 * sqrt(2), 7 * pi / 12)
    assert _egaux(forme_algebrique(*produit), (sqrt(3) - I) * (-1 + I))


def test_les_exemples_5_et_6_par_la_formule_de_moivre() -> None:
    assert puissance_polaire(PREMIER, 4) == (16, -2 * pi / 3)
    assert puissance_polaire(SECOND, 5) == (4 * sqrt(2), 15 * pi / 4)


def test_l_exemple_7_le_quotient_retranche_les_arguments() -> None:
    quotient = quotient_polaire(
        puissance_polaire(PREMIER, 4), puissance_polaire(SECOND, 5)
    )
    assert quotient == (2 * sqrt(2), -53 * pi / 12)
    attendu = (sqrt(3) - I) ** 4 / (-1 + I) ** 5
    assert _egaux(forme_algebrique(*quotient), attendu)


def test_l_inverse_inverse_le_module_et_oppose_l_argument() -> None:
    assert inverse_polaire(PREMIER) == (R(1, 2), pi / 6)
    assert _egaux(forme_algebrique(*inverse_polaire(PREMIER)), 1 / (sqrt(3) - I))


def test_un_module_negatif_est_refuse() -> None:
    with pytest.raises(ValueError, match="module negatif"):
        forme_algebrique(-1, pi)


def test_zero_n_a_pas_d_inverse() -> None:
    with pytest.raises(ValueError, match="module nul"):
        inverse_polaire((sp.Integer(0), pi))


def test_on_ne_divise_pas_par_zero() -> None:
    with pytest.raises(ValueError, match="module nul"):
        quotient_polaire(PREMIER, (sp.Integer(0), sp.Integer(0)))


def test_le_lieu_de_la_page_11_est_un_cercle() -> None:
    lieu = lieu_dans_le_plan_de_gauss(lambda z: sp.Abs(z - 5 * I) ** 2 - 9)
    assert cercle_du_lieu(lieu) == ((0, 5), 3)


def test_un_cercle_donne_avec_un_facteur_commun() -> None:
    x, y = sp.symbols("x y", real=True)
    assert cercle_du_lieu(2 * x**2 + 2 * y**2 - 4 * x - 6) == ((1, 0), 2)


def test_une_droite_n_est_pas_un_cercle() -> None:
    x, y = sp.symbols("x y", real=True)
    with pytest.raises(ValueError, match="pas un cercle"):
        cercle_du_lieu(x + y - 1)


def test_des_carres_de_coefficients_differents_ne_font_pas_un_cercle() -> None:
    x, y = sp.symbols("x y", real=True)
    with pytest.raises(ValueError, match="pas un cercle"):
        cercle_du_lieu(x**2 + 2 * y**2 - 1)


def test_un_terme_croise_ne_fait_pas_un_cercle() -> None:
    x, y = sp.symbols("x y", real=True)
    with pytest.raises(ValueError, match="pas un cercle"):
        cercle_du_lieu(x**2 + y**2 + x * y - 1)


def test_un_rayon_imaginaire_est_refuse() -> None:
    x, y = sp.symbols("x y", real=True)
    with pytest.raises(ValueError, match="aucun point"):
        cercle_du_lieu(x**2 + y**2 + 1)
