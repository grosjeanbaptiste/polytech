"""Les formules de poussee sont *integrees*, pas transcrites."""

import sympy as sp

from polytech.fluides.domain.grandeurs import Hauteur, MasseVolumique
from polytech.fluides.domain.statique.parements import (
    ParementTriangulaire,
    ParementVertical,
)
from polytech.fluides.domain.statique.symboles import ALPHA, RHO, G, H, L

EAU = MasseVolumique(kg_par_m3=1_000.0)
MUR = ParementVertical(fluide=EAU, largeur=1.0)
VALLEE = ParementTriangulaire(fluide=EAU, pente=0.5)


def _egal(obtenu: sp.Expr, attendu: sp.Expr) -> bool:
    return bool(sp.simplify(obtenu - attendu) == 0)


def test_le_parement_vertical_integre_en_un_demi_rho_g_l_h_carre() -> None:
    assert _egal(MUR.expression_poussee(), RHO * G * L * H**2 / 2)


def test_le_parement_vertical_a_un_moment_en_h_cube() -> None:
    assert _egal(MUR.expression_moment(), RHO * G * L * H**3 / 6)


def test_la_vallee_en_v_integre_en_alpha_rho_g_h_cube_sur_trois() -> None:
    assert _egal(VALLEE.expression_poussee(), ALPHA * RHO * G * H**3 / 3)


def test_l_exposant_de_la_poussee_est_deduit_de_l_integrale() -> None:
    assert MUR.exposant_poussee() == 2
    assert VALLEE.exposant_poussee() == 3


def test_les_rapports_sont_des_fractions_exactes() -> None:
    """Le corrige affiche 1/4 et 1/8 : des rationnels, pas des flottants."""
    assert MUR.rapport_poussee(sp.Rational(1, 2)) == sp.Rational(1, 4)
    assert MUR.rapport_moment(sp.Rational(1, 2)) == sp.Rational(1, 8)
    assert VALLEE.rapport_poussee(sp.Rational(1, 2)) == sp.Rational(1, 2)


def test_la_valeur_numerique_decoule_de_la_meme_integrale() -> None:
    """Aucune formule n'est ecrite deux fois : le nombre vient de l'integrale."""
    attendu = 0.5 * 1000 * 9.81 * 1.0 * 10.0**2
    assert MUR.poussee(Hauteur(metres=10.0)) == attendu
