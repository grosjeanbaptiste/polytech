"""Le tenseur d'inertie, sa rotation, le theoreme de Steiner, les pendules.

Source : `MecaniqueRationnelle2/SYLLABUS/Syllabus_Theorie_2025_2026.pdf`,
chapitre 6 -- pages 104 et 105 (6.7) a (6.11), page 109 (6.35), pages 110 et
111 (6.44) a (6.49), pages 113 et 114 (6.58) a (6.60), page 116 (6.63).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.mecanique.domain.inertie.oscillateurs import (
    longueur_du_pendule_simple_isochrone,
    periode_de_torsion,
    periode_du_pendule_compose,
)
from polytech.mecanique.domain.inertie.repartition import Repartition
from polytech.mecanique.domain.inertie.solides import brique, tige
from polytech.mecanique.domain.inertie.tenseur import (
    angle_principal,
    rotation_autour_de_z,
    steiner,
    steiner_pour_un_axe,
    tenseur_d_inertie,
    tourne,
)
from polytech.mecanique.domain.pesanteur import PESANTEUR

M, A, B, C, L, G = sp.symbols("m a b c L g", positive=True)
THETA = sp.Symbol("theta", real=True)
COIN = brique(M, (A, B, C), origine=(0, 0, 0))


def test_le_tenseur_porte_les_moments_et_l_oppose_des_produits() -> None:
    """Page 109, (6.35) : `I_11 = I_x`, `I_12 = -P_xy`, et il est symetrique."""
    tenseur = tenseur_d_inertie(COIN)
    assert sp.simplify(tenseur[0, 0] - M * (B**2 + C**2) / 3) == 0
    assert sp.simplify(tenseur[0, 1] + M * A * B / 4) == 0
    assert sp.simplify(tenseur[1, 2] + M * B * C / 4) == 0
    assert tenseur == tenseur.T


def test_la_rotation_des_axes_est_a_i_a_transposee() -> None:
    """Pages 110 et 111, (6.44) a (6.48), autour de `z`."""
    ix, iy, pxy = sp.symbols("I_x I_y P_xy", real=True)
    plan = sp.Matrix([[ix, -pxy, 0], [-pxy, iy, 0], [0, 0, ix + iy]])
    tourne_ = tourne(plan, rotation_autour_de_z(THETA))
    s, c = sp.sin(THETA), sp.cos(THETA)
    assert sp.simplify(tourne_[0, 0] - (c**2 * ix + s**2 * iy - 2 * s * c * pxy)) == 0
    assert sp.simplify(tourne_[1, 1] - (s**2 * ix + c**2 * iy + 2 * s * c * pxy)) == 0
    hors = s * c * (iy - ix) - (c**2 - s**2) * pxy
    assert sp.simplify(tourne_[0, 1] - hors) == 0


def test_l_angle_principal_annule_le_produit_d_inertie() -> None:
    """Page 111, (6.49) : `tan(2 theta*) = 2 P_xy / (I_y - I_x)`."""
    plan = sp.Matrix([[4, -1, 0], [-1, 2, 0], [0, 0, 6]])
    angle = angle_principal(sp.Integer(4), sp.Integer(2), sp.Integer(1))
    assert sp.simplify(sp.tan(2 * angle) + 1) == 0
    assert sp.simplify(tourne(plan, rotation_autour_de_z(angle))[0, 1]) == 0
    with pytest.raises(ValueError, match="moments egaux et produit nul"):
        angle_principal(sp.Integer(3), sp.Integer(3), sp.Integer(0))


def test_steiner_ramene_le_tenseur_du_centre_de_masse_au_point() -> None:
    """Page 113, (6.58) : `I_O = I_G + m (a^2 delta - a_alpha a_beta)`."""
    au_centre = tenseur_d_inertie(brique(M, (A, B, C)))
    transporte = steiner(au_centre, M, COIN.centre_de_masse())
    assert sp.simplify(transporte - tenseur_d_inertie(COIN)) == sp.zeros(3, 3)


def test_steiner_pour_un_axe_ajoute_la_masse_fois_la_distance_au_carre() -> None:
    """Page 114, (6.60) : la tige, `m L^2 / 12 + m (L / 2)^2 = m L^2 / 3`."""
    assert tige(M, L).centre_de_masse() == (L / 2, 0, 0)
    assert sp.simplify(steiner_pour_un_axe(M * L**2 / 12, M, L / 2) - M * L**2 / 3) == 0


def test_la_periode_du_pendule_compose_et_la_condition_de_huygens() -> None:
    """Pages 104 et 105, (6.7), (6.10), (6.11) : la tige vaut un pendule de 2L/3."""
    inertie = M * L**2 / 3
    isochrone = longueur_du_pendule_simple_isochrone(inertie, M, L / 2)
    assert sp.simplify(isochrone - 2 * L / 3) == 0
    compose = periode_du_pendule_compose(inertie, M, L / 2, G)
    assert sp.simplify(compose - 2 * sp.pi * sp.sqrt(isochrone / G)) == 0
    reel = periode_du_pendule_compose(sp.Float(1) / 3, sp.Float(1), sp.Float(0.5))
    assert float(reel) == pytest.approx(
        2 * 3.141592653589793 * (2 / 3 / PESANTEUR) ** 0.5
    )


def test_la_periode_de_l_oscillateur_de_torsion() -> None:
    """Page 116, (6.63) : `T = 2 pi racine(I / K)`."""
    assert periode_de_torsion(sp.Integer(8), sp.Integer(2)) == 4 * sp.pi
    with pytest.raises(ValueError, match="raideur de torsion nulle"):
        periode_de_torsion(sp.Integer(4), sp.Integer(0))


def test_une_repartition_sans_masse_n_a_pas_de_centre() -> None:
    """Le centre de masse divise par la masse, relation (6.5)."""
    x = sp.Symbol("x", real=True)
    vide = Repartition((x, sp.Integer(0), sp.Integer(0)), sp.Integer(0), ((x, 0, 1),))
    with pytest.raises(ValueError, match="repartition sans masse"):
        vide.centre_de_masse()
    assert tige(M, L).masse() == M


def test_un_moment_est_l_integrale_d_un_poids_sur_la_masse() -> None:
    """Page 103, (6.5) et (6.6) : `m L / 2` a l'ordre un, `m L^2 / 3` a l'ordre deux."""
    barre = tige(M, L)
    abscisse = barre.position[0]
    assert sp.simplify(barre.integrale(abscisse) - M * L / 2) == 0
    assert sp.simplify(barre.integrale(abscisse**2) - M * L**2 / 3) == 0


def test_un_pendule_suspendu_par_son_centre_de_masse_n_oscille_pas() -> None:
    """(6.10) : `r_G = 0` annule le moment de rappel."""
    with pytest.raises(ValueError, match="centre de masse sur l'axe"):
        longueur_du_pendule_simple_isochrone(M * L**2, M, sp.Integer(0))
