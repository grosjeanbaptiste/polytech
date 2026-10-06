"""Electromagnetisme : l'exercice cote du 16 mai 2019, cinq questions.

Source : `Electromagnetisme/EXS/Exs_Cote_2018_2019_Solution.pdf`, onze
pages -- enonce imprime, solutions manuscrites --, transcrit dans
`transcriptions/Electromagnetisme/EXS/Exs_Cote_2018_2019_Solution.md`.

Chaque resultat du manuscrit est recopie comme oracle et recoupe par un
autre chemin : le potentiel somme sur les quatre fils, le flux derive
morceau par morceau, la boucle d'Ampere refermee, Millman, la valeur
finale des images de Laplace.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.application import cote_2019_fermeture as fe
from polytech.electromagnetisme.application import cote_2019_fils as fi
from polytech.electromagnetisme.application import cote_2019_noyau as ny
from polytech.electromagnetisme.application import cote_2019_spire as sr
from polytech.electromagnetisme.application import cote_2019_thevenin as th

# --- Question 1 : deux fils l'un au-dessus de l'autre, devant un plan -------


def test_q1_le_coefficient_mutuel_du_manuscrit() -> None:
    """Page 2 : `K12 L = ln((h1 + h2)/(h1 - h2)) / (2 pi eps0)`."""
    attendu = sp.log((fi.H1 + fi.H2) / (fi.H1 - fi.H2)) / (2 * sp.pi * fi.EPS)
    assert sp.simplify(fi.coefficient_mutuel() - attendu) == 0


def test_q1_le_coefficient_propre() -> None:
    """Page 2 : le premier terme de `V(1)`, `ln(2 h1 / R) / (2 pi eps0)`."""
    attendu = sp.log(2 * fi.H1 / fi.RAYON) / (2 * sp.pi * fi.EPS)
    assert sp.simplify(fi.coefficient_propre() - attendu) == 0


def test_q1_la_hauteur_du_second_conducteur() -> None:
    """Page 2 : `h2 = 4,5` m."""
    assert fi.hauteur() == pytest.approx(4.5, abs=5e-3)
    retour = math.log((5 + fi.hauteur()) / (5 - fi.hauteur())) / (
        2 * math.pi * fi.PERMITTIVITE_DU_MANUSCRIT
    )
    assert retour == pytest.approx(53e9, rel=1e-9, abs=0)


def test_q1_le_champ_du_manuscrit() -> None:
    """Page 3 : 2397,81 V/m selon `+u_y`, avec `h2 = 4,5` m et `eps0 = 8,85e-12`."""
    horizontal, vertical = fi.champ_en_p(hauteur=4.5)
    assert horizontal == 0
    assert vertical == pytest.approx(2397.81, abs=5e-3)


def test_q1_le_champ_est_sensible_a_l_arrondi_de_la_hauteur() -> None:
    """Avec la hauteur non arrondie le champ baisse de pres d'un demi pour cent."""
    arrondi = fi.champ_en_p(hauteur=4.5)[1]
    exact = fi.champ_en_p(hauteur=fi.hauteur())[1]
    assert 0.002 < (arrondi - exact) / arrondi < 0.006


# --- Question 2 : une spire dans une induction en dents de scie -------------


def test_q2_le_flux_et_la_force_electromotrice() -> None:
    """Page 4 : `phi = -a^2 sin(alpha) B(t)`, `E12 = a^2 sin(alpha) dB/dt`."""
    induction = sp.Function("B")(sr.T)
    flux, fem = sr.flux_et_fem(induction)
    assert sp.simplify(flux + sr.COTE**2 * sp.sin(sr.ALPHA) * induction) == 0
    assert sp.simplify(fem - sr.COTE**2 * sp.sin(sr.ALPHA) * induction.diff(sr.T)) == 0


def test_q2_les_quatre_paliers() -> None:
    """Page 5 : `+2`, `-2`, `+2`, `-2` fois `a^2 sin(alpha)`, puis zero."""
    unite = sr.COTE**2 * sp.sin(sr.ALPHA)
    paliers = sr.paliers()
    assert [sp.simplify(valeur / unite) for _, _, valeur in paliers] == [
        2,
        -2,
        2,
        -2,
        0,
    ]
    assert [(debut, fin) for debut, fin, _ in paliers][:4] == [
        (0, sp.Rational(1, 2)),
        (sp.Rational(1, 2), 1),
        (1, sp.Rational(3, 2)),
        (sp.Rational(3, 2), 2),
    ]


def test_q2_l_induction_revient_a_zero() -> None:
    """L'aire sous `E12` est nulle sur chaque triangle : le flux revient a zero."""
    assert sr.induction(0) == sr.induction(1) == sr.induction(2) == 0
    assert sr.induction(sp.Rational(1, 2)) == sr.induction(sp.Rational(3, 2)) == 1
    unite = sr.COTE**2 * sp.sin(sr.ALPHA)
    aire = sum(valeur * (fin - debut) for debut, fin, valeur in sr.paliers()[:4])
    assert sp.simplify(aire / unite) == 0


# --- Question 3 : un noyau a deux entrefers ---------------------------------


def test_q3_l_induction_et_les_flux() -> None:
    """Page 7 : 0,88 T, `3,52e-4` Wb par colonne, `7,04e-4` Wb au centre."""
    etat = ny.etat()
    assert etat.induction == pytest.approx(0.88, abs=5e-3)
    assert etat.flux_lateral == pytest.approx(3.52e-4, rel=2e-3, abs=0)
    assert etat.flux_central == pytest.approx(7.04e-4, rel=2e-3, abs=0)
    assert etat.induction_centrale == pytest.approx(etat.induction, rel=1e-12, abs=0)


def test_q3_les_amperes_tours_du_manuscrit() -> None:
    """Page 7 : `160 x 0,08 + 160 x 0,218 + 700e3 x 2e-3 = 1447,68` A."""
    centre, cote, air = ny.amperes_tours()
    assert centre == pytest.approx(12.8, rel=1e-12, abs=0)
    assert cote == pytest.approx(34.88, rel=1e-12, abs=0)
    assert air == pytest.approx(1400.0, rel=1e-12, abs=0)
    assert centre + cote + air == pytest.approx(1447.68, rel=1e-12, abs=0)


def test_q3_le_nombre_de_spires() -> None:
    """Page 7 : `N = 4534,7`, donc 4535, avec le flux arrondi du manuscrit."""
    assert ny.spires(flux_central=7.04e-4) == pytest.approx(4534.7, abs=0.05)
    assert round(ny.spires(flux_central=7.04e-4)) == 4535


def test_q3_sans_arrondir_l_induction_on_trouve_une_spire_de_plus() -> None:
    """`B = mu_0 H_e` vaut 0,8796 T et non 0,88 : `N = 4535,6`."""
    assert ny.etat().induction == pytest.approx(0.8796, abs=5e-5)
    assert ny.spires() == pytest.approx(4535.6, abs=0.05)


def test_q3_la_lecture_de_la_courbe_pese_quelques_spires() -> None:
    """Dix amperes par metre de plus a la lecture : cinq spires de plus."""
    ecart = ny.spires(champ=170.0) - ny.spires(champ=160.0)
    assert ecart == pytest.approx(4.7, abs=0.1)


def test_q3_le_courant_et_l_inductance_se_referment() -> None:
    """`I = N I / N`, puis `L = N Phi_c / I` : on retrouve les dix henrys."""
    spires = ny.spires()
    courant = sum(ny.amperes_tours()) / spires
    assert spires * ny.etat().flux_central / courant == pytest.approx(
        10.0, rel=1e-12, abs=0
    )
    assert courant == pytest.approx(0.319, abs=5e-4)


# --- Question 4 : Thevenin a deux sources de tension et une de courant -----


def test_q4_le_generateur_equivalent() -> None:
    """Page 8 et 9 : `R_TH = 1/3` ohm, `E_TH = 31/3` V."""
    tension, resistance = th.thevenin()
    assert resistance == sp.Rational(1, 3)
    assert tension == sp.Rational(31, 3)


def test_q4_les_courants_de_mailles_du_manuscrit() -> None:
    """Page 9 : `I2 = 29/3` A, d'ou `E_TH = E1 + E2 - R4 I2`."""
    premier, second = th.courants_de_mailles()
    assert second == sp.Rational(29, 3)
    assert premier == 2 * second - 10
    assert 10 + 10 - second == th.thevenin()[0]


def test_q4_la_resistance_shuntee_ne_compte_pas() -> None:
    """`R2` est en parallele sur la bobine : sa valeur est indifferente."""
    assert th.thevenin(r2=sp.Integer(1000)) == th.thevenin()


def test_q4_le_court_circuit() -> None:
    """`I_cc = E_TH / R_TH = 31` A, mesure sur le reseau."""
    assert th.courant_de_court_circuit() == 31


# --- Question 5 : fermeture sur une branche R-C -----------------------------


def test_q5_les_conditions_initiales() -> None:
    """Page 11 : `i(0) = 0`, `i1(0) = E/(R1 + R2)`, `u_C(0) = 0`."""
    assert fe.conditions_initiales() == (0, fe.E / (fe.R1 + fe.R2), 0)


def test_q5_le_systeme_du_manuscrit() -> None:
    """Page 11 : les equations (1), (2) et (3), conditions initiales reportees."""
    p, courant, autre, tension = fe.P, fe.IP, fe.I1P, fe.UC
    un, deux, trois = fe.systeme()
    depart = fe.E / (fe.R1 + fe.R2)
    attendue_1 = (
        (fe.R1 + p * fe.L1) * courant
        + (fe.R1 + fe.R2 + p * (fe.L1 + fe.L2)) * autre
        - fe.E / p
        - (fe.L1 + fe.L2) * depart
    )
    attendue_2 = (
        fe.R3 * courant - (fe.R2 + p * fe.L2) * autre + tension + fe.L2 * depart
    )
    attendue_3 = -courant + p * fe.C * tension
    assert sp.simplify(un - attendue_1) == 0
    assert sp.simplify(deux - attendue_2) == 0
    assert sp.simplify(trois - attendue_3) == 0


def test_q5_le_regime() -> None:
    """Page 11 : `i = 0`, `u_C = R2 E / (R1 + R2)`."""
    courant, tension = fe.regime()
    assert courant == 0
    assert sp.simplify(tension - fe.R2 * fe.E / (fe.R1 + fe.R2)) == 0


def test_q5_le_courant_part_de_zero() -> None:
    """Les deux bobines interdisent tout saut : `i(0+) = 0`.

    Sa pente, elle, est fixee par les deux bobines :
    `(L1 + L2) R2 E / (L1 L2 (R1 + R2))`.
    """
    depart, pente = fe.depart_du_courant()
    assert depart == 0
    attendue = (fe.L1 + fe.L2) * fe.R2 * fe.E / (fe.L1 * fe.L2 * (fe.R1 + fe.R2))
    assert sp.simplify(pente - attendue) == 0
