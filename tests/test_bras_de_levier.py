"""Le bras de levier se deduit de la ligne d'action, il ne se postule pas.

``expression_moment`` multipliait la poussee par ``H / 3`` -- vrai pour un
parement vertical, faux des que la largeur mouillee varie, car la ligne
d'action remonte alors. Le bras est desormais derive.
"""

import pytest
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


def test_le_bras_du_mur_vertical_vaut_le_tiers_de_la_hauteur() -> None:
    """H - 2H/3 : le resultat connu, mais obtenu et non pose."""
    assert _egal(MUR.expression_bras_de_levier(), H / 3)


def test_le_bras_de_la_vallee_en_v_vaut_la_moitie_de_la_hauteur() -> None:
    """La ligne d'action remonte a H/2 : le bras vaut H/2, pas H/3."""
    assert _egal(VALLEE.expression_bras_de_levier(), H / 2)


def test_le_moment_du_mur_vertical_est_inchange() -> None:
    """Non-regression : la valeur publiee par le corrige ne bouge pas."""
    assert _egal(MUR.expression_moment(), RHO * G * L * H**3 / 6)


def test_le_moment_de_la_vallee_en_v_utilise_son_propre_bras() -> None:
    """Avec H/3 on aurait obtenu un quart de moins que la verite."""
    assert _egal(VALLEE.expression_moment(), ALPHA * RHO * G * H**4 / 6)


def test_la_hauteur_de_la_ligne_d_action_suit_la_geometrie() -> None:
    """Evaluation numerique : le mur donne H/3, la vallee H/2."""
    assert MUR.hauteur_ligne_action(Hauteur(metres=9.0)) == pytest.approx(3.0)
    assert VALLEE.hauteur_ligne_action(Hauteur(metres=9.0)) == pytest.approx(4.5)


def test_la_ligne_d_action_est_definie_par_une_egalite_de_moments() -> None:
    """h_a F = int h dF : la definition, dont le quotient est la resolution."""
    equation = MUR.equation_ligne_action()
    resolue = equation.subs(
        {
            sp.Symbol("h_a"): MUR.expression_ligne_action(),
            sp.Symbol("F"): MUR.expression_poussee(),
        },
    )
    # Sympy reduit l'egalite a `true` des qu'elle est satisfaite : c'est
    # exactement la verification voulue.
    assert resolue is sp.true
