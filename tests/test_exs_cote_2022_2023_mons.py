"""Electromagnetisme : l'exercice cote du 16 mai 2023 (Mons), trois questions.

Source : `Electromagnetisme/EXS/Exs_Cote_2022_2023_Mons_Solution.pdf`, six
pages -- enonce imprime, solutions manuscrites --, transcrit dans
`transcriptions/Electromagnetisme/EXS/Exs_Cote_2022_2023_Mons_Solution.md`.

Chaque resultat du manuscrit est recopie comme oracle et recoupe par un
autre chemin : le diviseur de tension, la force de Lorentz sur les deux
cotes du cadre, le bilan de puissance du primaire.
"""

from __future__ import annotations

import cmath
import math

import pytest
import sympy as sp

from polytech.electromagnetisme.application import cote_2023_mons_bobines as bo
from polytech.electromagnetisme.application import cote_2023_mons_cadre as ca
from polytech.electromagnetisme.application import cote_2023_mons_thevenin as th

# --- Question 1 : Thevenin en regime ----------------------------------------


def test_q1_le_generateur_equivalent() -> None:
    """Pages 1 et 2 : `R_T = 4/9` ohm (0,44), `E_T = 40/9` V (4,44)."""
    tension, resistance = th.thevenin()
    assert resistance == sp.Rational(4, 9)
    assert tension == sp.Rational(40, 9)


def test_q1_par_le_diviseur_du_manuscrit() -> None:
    """Page 2 : `R14 = 4/5` ohm, puis `E_T = R14/(R0 + R14) E0`."""
    assert th.resistances_groupees() == sp.Rational(4, 5)
    assert th.tension_par_le_diviseur() == sp.Rational(40, 9)


def test_q1_le_court_circuit() -> None:
    """Second chemin : `A` et `B` relies, il ne reste que `E0/R0`."""
    tension, resistance = th.thevenin()
    assert th.court_circuit() == 10
    assert tension / resistance == th.court_circuit()


def test_q1_ni_r2_ni_r3_ne_comptent() -> None:
    """La bobine court-circuite `R2`, le condensateur ouvre `R3`."""
    assert th.thevenin(r2=50, r3=70) == th.thevenin()


# --- Question 2 : un cadre dans un champ qui croit avec x -------------------


def test_q2_le_flux_du_manuscrit() -> None:
    """Page 4 : `phi = -B0 a b - (k b/2)(a^2 + 2 a d1)`."""
    attendu = -ca.B0 * ca.A * ca.B - ca.K * ca.B / 2 * (ca.A**2 + 2 * ca.A * ca.D1)
    assert sp.simplify(ca.flux() - attendu) == 0


def test_q2_le_flux_ne_depend_pas_de_d2() -> None:
    """Le champ ne varie pas selon `y` : la cote `d2` disparait."""
    assert ca.D2 not in ca.flux().free_symbols


def test_q2_la_force_electromotrice_du_manuscrit() -> None:
    """Page 4 : `E = k b a v`."""
    assert sp.simplify(ca.fem() - ca.K * ca.B * ca.A * ca.V) == 0


def test_q2_par_la_force_de_lorentz() -> None:
    """Second chemin : `v b` fois l'ecart du champ entre les deux cotes verticaux."""
    assert sp.simplify(ca.fem_par_les_cotes() - ca.fem()) == 0
    assert ca.fem_par_les_cotes(ca.D1 + ca.V * ca.T) == ca.fem_par_les_cotes()


def test_q2_un_champ_uniforme_n_induit_rien() -> None:
    """`k = 0` : le flux ne change pas quand le cadre glisse."""
    assert ca.fem().subs(ca.K, 0) == 0


# --- Question 3 : trois mailles couplees ------------------------------------


def test_q3_les_signes_des_deux_mutuelles() -> None:
    """`L12` : deux entrees par le point. `L34` : une seule."""
    assert bo.signes() == (1, -1)


def test_q3_la_matrice_du_manuscrit() -> None:
    """Page 6 : la matrice de Laplace, conditions initiales nulles."""
    p = bo.P
    attendue = sp.Matrix(
        [
            [bo.R1 + p * bo.L1, p * bo.L12, 0],
            [p * bo.L12, bo.R2 + p * (bo.L2 + bo.L3), -p * bo.L34],
            [0, -p * bo.L34, bo.R3 + p * (bo.L4 + bo.L5)],
        ]
    )
    matrice, second_membre = bo.systeme()
    assert sp.simplify(matrice - attendue) == sp.zeros(3, 3)
    assert second_membre == sp.Matrix([bo.E / p, 0, 0])


def test_q3_le_regime_du_manuscrit() -> None:
    """Page 6 : `I1 = E/R1`, `I2 = 0` -- et `I3 = 0`."""
    assert bo.regime() == (bo.E / bo.R1, 0, 0)


def test_q3_le_courant_du_secondaire() -> None:
    """Page 6 : `I2 = -29,55 - 39,55 j = 49,37 e^(-2,21 j)` A."""
    courant = bo.courant_du_secondaire()
    assert courant.real == pytest.approx(-29.55, abs=5e-3)
    assert courant.imag == pytest.approx(-39.55, abs=5e-3)
    assert abs(courant) == pytest.approx(49.37, abs=5e-3)
    assert cmath.phase(courant) == pytest.approx(-2.21, abs=5e-3)


def test_q3_l_amplitude_du_manuscrit() -> None:
    """Page 6 : `i2(t) = 69,82 cos(100 pi t - 2,21)` A."""
    amplitude, phase = bo.sinusoide()
    assert amplitude == pytest.approx(69.82, abs=5e-3)
    assert phase == pytest.approx(-2.21, abs=5e-3)


def test_q3_la_phase_en_degres() -> None:
    """Le manuscrit ecrit `126,6` degres ; `2,2125` rad en font `126,77`."""
    _, phase = bo.sinusoide()
    assert math.degrees(-phase) == pytest.approx(126.77, abs=5e-3)
    assert math.degrees(2.21) == pytest.approx(126.6, abs=5e-2)


def test_q3_la_maille_du_primaire_est_verifiee() -> None:
    """Second chemin : le courant trouve, reporte dans la maille, rend `E`."""
    assert bo.residu_du_primaire() == pytest.approx(0, abs=1e-12)


def test_q3_le_bilan_de_puissance_du_primaire() -> None:
    """La source fournit 10 W, `R1` en dissipe 13 : le secondaire en rendrait 3."""
    fournie, dissipee, transmise = bo.puissances()
    assert fournie == pytest.approx(10)
    assert dissipee == pytest.approx(13)
    assert transmise == pytest.approx(-3)
    assert fournie == pytest.approx(dissipee + transmise)


def test_q3_l_impedance_d_entree_est_sous_r1() -> None:
    """`E/I1 = 0,77 - 1,15 j` ohm : sa partie reelle est inferieure a `R1`."""
    impedance = bo.impedance_d_entree()
    assert impedance.real == pytest.approx(10 / 13)
    assert impedance.imag == pytest.approx(-15 / 13)
    assert impedance.real < bo.RESISTANCE
