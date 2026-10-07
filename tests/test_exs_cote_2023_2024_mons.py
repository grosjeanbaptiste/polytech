"""Electromagnetisme : l'exercice cote du 23 mai 2024 (Mons), trois questions.

Source : `Electromagnetisme/EXS/Exs_Cote_2023_2024_Mons_Solution.pdf`, six
pages -- enonce imprime, solutions manuscrites --, transcrit dans
`transcriptions/Electromagnetisme/EXS/Exs_Cote_2023_2024_Mons_Solution.md`.

Chaque resultat du manuscrit est recopie comme oracle et recoupe par un
autre chemin : le reseau resolu en entier, la force electromotrice de la
tige mobile, les theoremes de la valeur initiale et de la valeur finale,
le bilan d'energie.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.application import cote_2024_mons_commutation as co
from polytech.electromagnetisme.application import cote_2024_mons_thevenin as th
from polytech.electromagnetisme.application import cote_2024_mons_tiges as tg

# --- Question 1 : Thevenin, puis l'energie d'une capacite -------------------


def test_q1_le_generateur_equivalent() -> None:
    """Pages 1 et 2 : `R_T = 16/10 = 1,6` ohm, `E_T = -2` V."""
    tension, resistance = th.thevenin()
    assert resistance == sp.Rational(8, 5)
    assert tension == -2


def test_q1_la_formule_du_manuscrit() -> None:
    """Page 1 : `R_T = (R1 + R4 + R3) // R2`, soit `(2 + 6 + 8)/10`."""
    assert th.resistance_par_la_formule() == sp.Rational(16, 10)


def test_q1_le_courant_de_la_maille() -> None:
    """Page 2 : `I = -E/(R1 + R2 + R3 + R4) = -1` A, de A vers B dans `R2`."""
    assert th.courant_de_la_maille() == -1


def test_q1_r5_ne_compte_pas() -> None:
    """Le condensateur `C` ouvre la branche de `R5`."""
    assert th.thevenin(r5=50) == th.thevenin()


def test_q1_l_energie_de_la_capacite() -> None:
    """Page 2 : `(1/2) C0 Uc^2 = 4` microjoules."""
    assert th.energie_en_microjoules() == 4


def test_q1_la_capacite_branchee_sur_le_circuit() -> None:
    """Second chemin : `C0` sur le circuit d'origine porte bien `E_T`."""
    assert th.tension_de_la_capacite() == th.thevenin()[0]


# --- Question 2 : une tige qui glisse sur deux tiges en V -------------------


def test_q2_le_flux_du_manuscrit() -> None:
    """Page 4 : `phi(x') = (2/3) b x'^3 tg(alpha)`."""
    attendu = sp.Rational(2, 3) * tg.B * tg.X**3 * sp.tan(tg.ALPHA)
    assert sp.simplify(tg.flux() - attendu) == 0


def test_q2_la_force_electromotrice_du_manuscrit() -> None:
    """Page 4 : `|E| = 2 b v^3 t^2 tg(alpha)`, soit `2 b v d^2 tg(alpha)` en `d`."""
    dans_le_temps = 2 * tg.B * tg.V**3 * tg.T**2 * sp.tan(tg.ALPHA)
    assert sp.simplify(tg.fem_absolue() - dans_le_temps) == 0
    en_d = 2 * tg.B * tg.V * tg.D**2 * sp.tan(tg.ALPHA)
    assert sp.simplify(tg.fem_a_la_distance() - en_d) == 0


def test_q2_par_la_tige_mobile() -> None:
    """Second chemin : `B l v`, le champ `b d` sur une tige de `2 d tg(alpha)`."""
    assert sp.simplify(tg.fem_par_la_tige() - tg.fem_a_la_distance()) == 0


def test_q2_la_vitesse_du_manuscrit() -> None:
    """Page 4 : `v = 10/(2 x 0,1 x 5^2 x tg 20) = 5,49` m/s."""
    assert tg.vitesse() == pytest.approx(5.49, abs=5e-3)
    assert tg.vitesse() == pytest.approx(10 / (5 * math.tan(math.radians(20))))


def test_q2_la_longueur_de_la_tige() -> None:
    """En `x' = d` les deux tiges sont ecartees de `2 x 5 x tg 20 = 3,64` m."""
    assert tg.longueur_de_la_tige() == pytest.approx(3.64, abs=5e-3)


# --- Question 3 : une double commutation ------------------------------------


def test_q3_juste_avant_la_commutation() -> None:
    """Page 5 : `I1 = -E1/R12`, `I2 = E2/R2`, `Uc = E1`."""
    assert co.avant() == (-co.E1 / co.R12, co.E2 / co.R2, co.E1)


def test_q3_juste_avant_par_le_reseau() -> None:
    """Second chemin : le primaire resolu comme un reseau en regime."""
    courant, tension = co.avant_par_le_reseau()
    assert sp.simplify(courant - co.avant()[0]) == 0
    assert sp.simplify(tension - co.avant()[2]) == 0


def test_q3_le_signe_de_la_mutuelle() -> None:
    """`i1` monte dans `L1` et y entre par le point ; `i2` sort de `L2` par le sien."""
    assert co.signe() == -1


def test_q3_la_matrice_du_manuscrit() -> None:
    """Page 6 : trois equations, inconnues `I1`, `I2`, `Uc`."""
    p = co.P
    attendue = sp.Matrix(
        [
            [co.R12 + co.R11 + p * co.L1, -p * co.L12, 1],
            [-p * co.L12, co.R2 + p * co.L2, 0],
            [1, 0, -p * co.C],
        ]
    )
    matrice, _ = co.systeme()
    assert sp.simplify(matrice - attendue) == sp.zeros(3, 3)


def test_q3_le_second_membre_du_manuscrit() -> None:
    """Page 6 : les conditions initiales, passees a droite."""
    attendu = sp.Matrix(
        [
            -co.L1 * co.E1 / co.R12 - co.L12 * co.E2 / co.R2,
            co.L2 * co.E2 / co.R2 + co.L12 * co.E1 / co.R12,
            -co.C * co.E1,
        ]
    )
    _, second_membre = co.systeme()
    assert sp.simplify(second_membre - attendu) == sp.zeros(3, 1)


def test_q3_le_regime_du_manuscrit() -> None:
    """Page 6 : `I1 = 0`, `I2 = 0`, `Uc = 0`, par la valeur finale."""
    assert co.limites(sp.Integer(0)) == (0, 0, 0)


def test_q3_la_valeur_initiale_rend_les_conditions() -> None:
    """Second chemin : `lim p X(p)` a l'infini redonne l'etat d'avant."""
    obtenues = co.limites(sp.oo)
    for obtenue, attendue in zip(obtenues, co.avant(), strict=True):
        assert sp.simplify(obtenue - attendue.subs(co.EXEMPLE)) == 0


def test_q3_le_bilan_d_energie() -> None:
    """La derivee de l'energie stockee est l'oppose de la puissance dissipee."""
    assert co.residu_du_bilan_d_energie() == 0
