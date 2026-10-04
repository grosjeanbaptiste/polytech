"""TP 4 d'electronique : l'analyse nodale modifiee, six circuits.

Les matrices attendues sont recopiees de l'enonce (`TP_4.pdf`, pages 3 a 5)
et du corrige officiel (`TP_4_Solution.pdf`, pages 2 a 4) ; la
bibliotheque, elle, les *assemble* a partir de la liste des composants.
Deux chemins sans etape commune : l'un est ecrit a la main par l'auteur,
l'autre depose chaque element dans la matrice. Les tensions de sortie
sont comparees aux resultats que l'enonce attribue a « un logiciel de
calcul symbolique », recopies eux aussi.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.electronique.application import circuits_tp_04 as ct
from polytech.electronique.application import deductions_tp_04 as dd
from polytech.electronique.application import nombres_tp_04 as nb
from polytech.electronique.application.symboles_tp_04 import (
    COL,
    GM,
    GM1,
    GM2,
    IS,
    R1,
    R2,
    R3,
    R4,
    R5,
    REQ,
    RL,
    RO,
    RO1,
    RO2,
    RPI1,
    RPI2,
    RS,
    VS,
    VX,
    VY,
    C,
    R,
    S,
)


def _nul(expression: sp.Basic) -> bool:
    return bool(sp.simplify(expression) == sp.zeros(*expression.shape))


def test_circuit_1_a_la_matrice_du_corrige() -> None:
    """`TP_4_Solution.pdf`, page 2."""
    g1, g2, g3 = 1 / R1, 1 / R2, 1 / R3
    attendue = sp.Matrix([[-g1, g1, 1], [g1, -(g1 + g2 + g3), 0], [1, 0, 0]])
    matrice, membre = ct.systeme_du_circuit_1()
    assert _nul(matrice - attendue)
    assert _nul(membre - sp.Matrix([0, -IS, VS]))


def test_circuit_1_obeit_a_millman() -> None:
    """`V_2` est la moyenne de `V_S` et du courant injecte, ponderee."""
    v2 = (VS / R1 + IS) / (1 / R1 + 1 / R2 + 1 / R3)
    assert sp.simplify(ct.solution_du_circuit_1()[1] - v2) == 0


def test_circuit_2_a_la_matrice_du_corrige() -> None:
    """`TP_4_Solution.pdf`, page 3."""
    r1, r2, r3, r4, r5 = R1, R2, R3, R4, R5
    g1, g2, g3, g4, g5 = (1 / r for r in (r1, r2, r3, r4, r5))
    attendue = sp.Matrix(
        [
            [-g1, g1, 0, 0, 1],
            [g1, -(g1 + g2 + g4), g2, 0, 0],
            [0, g2, -(g2 + g3 + g5), g3, 0],
            [0, 0, g3, -g3, 0],
            [1, 0, 0, 0, 0],
        ]
    )
    matrice, membre = ct.systeme_du_circuit_2()
    assert _nul(matrice - attendue)
    assert _nul(membre - sp.Matrix([0, 0, 0, -IS, VS]))


def test_circuit_2_sans_source_de_courant_est_un_double_diviseur() -> None:
    """`I_S = 0` : aucun courant dans `R_3`, `V_4 = V_3`."""
    solution = ct.solution_du_circuit_2().subs(IS, 0)
    assert sp.simplify(solution[3] - solution[2]) == 0


def test_circuit_3_a_la_matrice_du_corrige() -> None:
    """`TP_4_Solution.pdf`, page 4."""
    g1, g2, g3 = 1 / R1, 1 / R2, 1 / R3
    attendue = sp.Matrix(
        [
            [-g1, g1, 0, 1, 0],
            [g1, -(g1 + g2), 0, 0, -1],
            [0, 0, -g3, 0, 1],
            [1, 0, 0, 0, 0],
            [0, -1, 1, 0, 0],
        ]
    )
    matrice, membre = ct.systeme_du_circuit_3()
    assert _nul(matrice - attendue)
    assert _nul(membre - sp.Matrix([IS, -IS, 0, VY, VX]))


def test_circuit_3_respecte_ses_deux_sources_de_tension() -> None:
    v1, v2, v3, _, _ = ct.solution_du_circuit_3()
    assert sp.simplify(v1 - VY) == 0
    assert sp.simplify(v3 - v2 - VX) == 0


def test_dephaseur_a_la_matrice_de_l_enonce() -> None:
    """`TP_4.pdf`, page 3, avec `Y = 1/R`."""
    y, cs = 1 / R, C * S
    attendue = sp.Matrix(
        [
            [-cs, cs, 0, 0, 1],
            [cs, -y - 2 * cs, cs, 0, 0],
            [0, cs, -y - 2 * cs, cs, 0],
            [0, 0, cs, -y - cs - 1 / RL, 0],
            [1, 0, 0, 0, 0],
        ]
    )
    assert _nul(ct.systeme_du_dephaseur()[0] - attendue)


def test_dephaseur_a_la_sortie_de_l_enonce() -> None:
    y = 1 / R
    attendue = (
        C**3 * S**3 / (C**3 * S**3 + 6 * C**2 * y * S**2 + 5 * C * y**2 * S + y**3)
    )
    assert sp.simplify(dd.transfert_du_dephaseur() - attendue) == 0


def test_dephaseur_inverse_a_la_pulsation_de_l_enonce() -> None:
    assert sp.simplify(dd.pulsation_d_inversion() - 1 / (sp.sqrt(6) * R * C)) == 0


def test_dephaseur_rend_un_vingt_neuvieme_change_de_signe() -> None:
    """Numeriquement, a `R = 1 kOhm`, `C = 1 nF` : un reel, et negatif."""
    w = 1 / (math.sqrt(6) * 1e3 * 1e-9)
    s = 1j * w
    b = 1 / (1e3 * 1e-9)
    h = s**3 / (s**3 + 6 * b * s**2 + 5 * b**2 * s + b**3)
    assert abs(h - (-1 / 29)) < 1e-12
    assert dd.gain_a_l_inversion() == sp.Rational(-1, 29)


def test_source_commune_a_la_matrice_de_l_enonce() -> None:
    """`TP_4.pdf`, page 3."""
    ro = RO
    attendue = sp.Matrix(
        [
            [-1 / RS, 1 / RS, 0, 0, 1],
            [1 / RS, -COL * S - 1 / RS, 0, COL * S, 0],
            [0, GM, -GM - 1 / R - 1 / ro, 1 / ro, 0],
            [0, COL * S - GM, GM + 1 / ro, -COL * S - 1 / REQ - 1 / ro, 0],
            [1, 0, 0, 0, 0],
        ]
    )
    assert _nul(ct.systeme_de_la_source_commune()[0] - attendue)


def test_source_commune_a_la_sortie_de_l_enonce() -> None:
    numerateur = REQ * (COL * S - GM + COL * R * GM * S)
    denominateur = (
        1 + R * GM + COL * S * (RS + REQ + R * RS * GM + R * REQ * GM + RS * REQ * GM)
    )
    assert (
        sp.simplify(dd.transfert_de_la_source_commune() - numerateur / denominateur)
        == 0
    )


def test_source_commune_sans_capacite() -> None:
    attendu = -REQ * GM / (1 + R * GM)
    assert sp.simplify(dd.gain_sans_capacite() - attendu) == 0


def test_source_commune_aux_valeurs_de_l_enonce() -> None:
    """Les nombres refaits a la main, sans la bibliotheque."""
    rs, r, gm, req, col = 50.0, 100.0, 0.05, 10e3, 1e-15
    nombres = nb.source_commune_numerique()
    assert math.isclose(nombres.gain_en_continu, -req * gm / (1 + r * gm))
    exact_infini = (
        req * (1 + r * gm) / (rs + req + r * rs * gm + r * req * gm + rs * req * gm)
    )
    assert math.isclose(nombres.gain_a_l_infini, exact_infini)
    assert math.isclose(nombres.zero, gm / (col * (1 + r * gm)))
    pole = -(1 + r * gm) / (
        col * (rs + req + r * rs * gm + r * req * gm + rs * req * gm)
    )
    assert math.isclose(nombres.pole, pole)


def test_source_commune_approchee_comme_le_corrige() -> None:
    """Le pole approche du corrige : `-(1 + 5) / (C_ol 500 (150 + 20))`."""
    nombres = nb.source_commune_numerique()
    assert math.isclose(nombres.pole_approche, -6 / (1e-15 * 500 * 170))
    assert math.isclose(nombres.gain_a_l_infini_approche, 120 / 170)
    assert math.isclose(nombres.capacite_miller, 500 * 1e-15)


def test_l_approximation_du_pole_tient_sous_le_pour_cent() -> None:
    nombres = nb.source_commune_numerique()
    assert abs(nombres.pole_approche / nombres.pole - 1) < 0.01


def test_cascode_a_la_matrice_de_l_enonce() -> None:
    """`TP_4.pdf`, page 5."""
    rp1, rp2 = RPI1, RPI2
    attendue = sp.Matrix(
        [
            [-1 / RS, 1 / RS, 0, 0, 1],
            [1 / RS, -1 / RS - 1 / rp2, 0, 0, 0],
            [0, -GM2, -GM1 - 1 / RO1 - 1 / RO2 - 1 / rp1, 1 / RO1, 0],
            [0, 0, GM1 + 1 / RO1, -1 / RL - 1 / RO1, 0],
            [1, 0, 0, 0, 0],
        ]
    )
    assert _nul(ct.systeme_de_la_cascode()[0] - attendue)


def test_cascode_a_les_sorties_de_l_enonce() -> None:
    denominateur = RL + RO1 + RO2 + GM1 * RO1 * RO2
    v3, v4 = dd.tensions_de_la_cascode()
    assert sp.simplify(v4 + VS * RL * GM2 * RO2 * (GM1 * RO1 + 1) / denominateur) == 0
    assert sp.simplify(v3 + VS * GM2 * RO2 * (RL + RO1) / denominateur) == 0


def test_cascode_a_vide() -> None:
    assert (
        sp.simplify(dd.gain_de_la_cascode_a_vide() + GM2 * RO2 * (GM1 * RO1 + 1)) == 0
    )


def test_cascode_aux_valeurs_de_l_enonce() -> None:
    """`R_L -> 0` : `V_3 / V_S = -0.02 * 20e3 * 20e3 / (40e3 + 0.02 * 4e8)`."""
    gm, ro = 0.02, 20e3
    nombres = nb.cascode_numerique()
    assert math.isclose(nombres.gain_interne, -gm * ro * ro / (2 * ro + gm * ro * ro))
    assert math.isclose(nombres.gain_a_vide, -gm * ro * (gm * ro + 1))
    assert math.isclose(nombres.facteur_miller, 1 - nombres.gain_interne)
