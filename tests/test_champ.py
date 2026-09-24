"""Cours III -- analyse d'un champ de vitesse (enonces B et C)."""

import pytest
import sympy as sp

from fluides.domain.cinematique.champ import ChampDeVitesse
from fluides.domain.cinematique.lignes import (
    FonctionDeCourantInexistante,
    PotentielInexistant,
)
from fluides.domain.cinematique.symboles import T, X, Y

CHAMP_B = ChampDeVitesse(u=X * T, v=X - Y)
CHAMP_C = ChampDeVitesse(u=2 * X, v=-2 * Y)


def _egal(obtenu: sp.Expr, attendu: sp.Expr) -> bool:
    return bool(sp.simplify(obtenu - attendu) == 0)


# --- Enonce B : u = x t, v = x - y -------------------------------------


def test_b_n_est_pas_stationnaire() -> None:
    assert CHAMP_B.est_stationnaire() is False


def test_b_a_une_divergence_egale_a_t_moins_un() -> None:
    assert _egal(CHAMP_B.divergence(), T - 1)
    assert CHAMP_B.est_incompressible() is False


def test_b_est_rotationnel_a_un_demi() -> None:
    assert _egal(CHAMP_B.taux_de_rotation(), sp.Rational(1, 2))
    assert CHAMP_B.est_rotationnel() is True


def test_b_a_le_champ_d_acceleration_attendu() -> None:
    ax, ay = CHAMP_B.acceleration()
    assert _egal(ax, X * (1 + T**2))
    assert _egal(ay, X * T - X + Y)


def test_b_acceleration_au_point_de_l_enonce() -> None:
    ax, ay = CHAMP_B.acceleration_en(x=1, y=2, t=2)
    assert (ax, ay) == pytest.approx((5.0, 3.0))


def test_b_se_deforme_avec_cisaillement() -> None:
    tenseur = CHAMP_B.tenseur_taux_deformation()
    assert _egal(tenseur[0, 0], T)
    assert _egal(tenseur[1, 1], sp.Integer(-1))
    assert _egal(tenseur[0, 1], sp.Rational(1, 2))


def test_b_n_admet_ni_fonction_de_courant_ni_potentiel() -> None:
    assert CHAMP_B.admet_une_fonction_de_courant() is False
    assert CHAMP_B.admet_un_potentiel() is False
    with pytest.raises(FonctionDeCourantInexistante):
        CHAMP_B.fonction_de_courant()
    with pytest.raises(PotentielInexistant):
        CHAMP_B.potentiel_des_vitesses()


# --- Enonce C : u = 2x, v = -2y ----------------------------------------


def test_c_est_stationnaire_incompressible_et_irrotationnel() -> None:
    assert CHAMP_C.est_stationnaire() is True
    assert CHAMP_C.est_incompressible() is True
    assert CHAMP_C.est_rotationnel() is False


def test_c_a_pour_fonction_de_courant_deux_xy() -> None:
    assert _egal(CHAMP_C.fonction_de_courant(), 2 * X * Y)


def test_c_a_pour_potentiel_x_carre_moins_y_carre() -> None:
    assert _egal(CHAMP_C.potentiel_des_vitesses(), X**2 - Y**2)


def test_c_etire_selon_x_et_aplatit_selon_y_a_aire_constante() -> None:
    tenseur = CHAMP_C.tenseur_taux_deformation()
    assert _egal(tenseur[0, 0], sp.Integer(2))
    assert _egal(tenseur[1, 1], sp.Integer(-2))
    assert _egal(tenseur[0, 1], sp.Integer(0))
    assert _egal(CHAMP_C.divergence(), sp.Integer(0))


def test_c_la_trajectoire_est_l_hyperbole_des_lignes_de_courant() -> None:
    """Ecoulement stationnaire : trajectoire et ligne de courant coincident."""
    xt, yt = CHAMP_C.trajectoire(x0=3, y0=5)
    assert _egal(sp.simplify(xt * yt), sp.Integer(15))


def test_c_acceleration_purement_convective() -> None:
    ax, ay = CHAMP_C.acceleration()
    assert _egal(ax, 4 * X)
    assert _egal(ay, 4 * Y)
