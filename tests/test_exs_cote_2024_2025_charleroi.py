"""Electromagnetisme : l'exercice cote du 12 mai 2025 (Charleroi), trois questions.

Sources : `Electromagnetisme/EXS/Exs_Cote_2024_2025_Charleroi_Enonce.pdf`
(six pages imprimees) et `..._Solution.pdf` (trois pages manuscrites),
transcrites dans
`transcriptions/Electromagnetisme/EXS/Exs_Cote_2024_2025_Charleroi_Solution.md`.

Chaque resultat du manuscrit est recopie comme oracle et recoupe par un
autre chemin. A la question 3 le recoupement **contredit** le manuscrit :
son courant change de sens entre ses deux croquis, pas sa valeur initiale.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.application import cote_2025_capteur as cp
from polytech.electromagnetisme.application import cote_2025_ouverture as ov
from polytech.electromagnetisme.application import cote_2025_thevenin as th

# --- Question 1 : Thevenin, puis un dipole actif ----------------------------


def test_q1_le_generateur_equivalent() -> None:
    """Page 1 : `R_TH = 1,5` ohm et `E_TH = -10,5` V."""
    tension, resistance = th.thevenin()
    assert resistance == sp.Rational(3, 2)
    assert tension == sp.Rational(-21, 2)


def test_q1_le_courant_de_maille_du_manuscrit() -> None:
    """Page 1 : `I1 = (-E2 - E1 + R2 I)/(R1 + R2 + R3) = -3,5` A."""
    assert th.courant_de_maille() == sp.Rational(-7, 2)


def test_q1_la_tension_du_dipole_par_l_equivalent() -> None:
    """Page 1 : `I2 = (E_TH + R_TH I')/(R_TH + R') = -1` A, `U = -6` V."""
    assert th.tension_par_l_equivalent() == -6


def test_q1_la_tension_du_dipole_par_le_circuit_entier() -> None:
    """Second chemin : le dipole branche sur le circuit d'origine, sans equivalent."""
    assert th.tension_par_le_circuit() == -6


def test_q1_le_condensateur_et_r4_ne_comptent_pas() -> None:
    """En regime la branche `C`-`R4` est ouverte : l'oter ne change rien."""
    assert th.thevenin(avec_la_branche_capacitive=False) == th.thevenin()


# --- Question 2 : une spire qui glisse sur un barreau -----------------------


def test_q2_le_flux_du_manuscrit() -> None:
    """Page 2 : `phi = 1,1 S - 0,1 S x_s`."""
    position = sp.Symbol("x")
    attendu = sp.Rational(11, 10) * cp.S - sp.Rational(1, 10) * cp.S * position
    assert sp.expand(cp.flux(position) - attendu) == 0


def test_q2_la_position_du_manuscrit() -> None:
    """Page 2 : `x_s(t) = 2.10^-2/(10 pi S) sin(10 pi t)`."""
    attendue = sp.Rational(2, 100) / (10 * sp.pi * cp.S) * sp.sin(10 * sp.pi * cp.T)
    assert sp.simplify(cp.position() - attendue) == 0


def test_q2_la_section_du_manuscrit() -> None:
    """Page 2 : `S = 2.10^-2/pi` m^2, soit `63,7` cm^2."""
    assert cp.section() == sp.Rational(2, 100) / sp.pi
    assert float(cp.section()) * 1e4 == pytest.approx(63.7, abs=0.05)


def test_q2_la_position_rend_bien_la_force_electromotrice_voulue() -> None:
    """Second chemin : reporter `x_s(t)` dans `-d(phi)/dt`."""
    assert sp.simplify(cp.fem_obtenue() - cp.FEM_VOULUE) == 0


def test_q2_la_spire_part_de_l_origine_et_atteint_dix_centimetres() -> None:
    position = cp.position().subs(cp.S, cp.section())
    assert position.subs(cp.T, 0) == 0
    assert position.subs(cp.T, sp.Rational(1, 20)) == sp.Rational(1, 10)


def test_q2_le_champ_reste_pres_de_vingt_amperes_par_metre() -> None:
    """La droite n'est donnee qu'autour de `H = 20` : la course l'en ecarte de 1 %."""
    assert cp.plage_du_champ() == (sp.Rational(99, 5), sp.Rational(101, 5))


def test_q2_la_vitesse_maximale() -> None:
    """`2.10^-2/S = pi` m/s : la spire va vite, pour deux millivolts."""
    assert cp.vitesse_maximale() == sp.pi


# --- Question 3 : ouvrir d'un cote, fermer de l'autre -----------------------


def test_q3_le_courant_d_avant_du_manuscrit() -> None:
    """Page 3 : module `E0 racine(2)/racine((R0+R1)^2 + (wL)^2)`, retard `arctan`."""
    somme = ov.R0 + ov.R1
    module = ov.E0 * sp.sqrt(2) / sp.sqrt(somme**2 + (ov.W * ov.L) ** 2)
    attendu = module * sp.cos(ov.W * ov.T - sp.atan(ov.W * ov.L / somme))
    valeurs = {ov.E0: 3, ov.W: 2, ov.R0: 5, ov.R1: 1, ov.L: 4, ov.T: sp.Rational(1, 3)}
    assert abs(sp.N((ov.courant_avant() - attendu).subs(valeurs), 30)) < 1e-25


def test_q3_le_courant_d_avant_verifie_l_equation_du_circuit() -> None:
    """Second chemin, sans phaseur : `e0 = (R0 + R1) i + L di/dt`."""
    avant = ov.courant_avant()
    source = ov.E0 * sp.sqrt(2) * sp.cos(ov.W * ov.T)
    reste = (ov.R0 + ov.R1) * avant + ov.L * avant.diff(ov.T) - source
    assert sp.simplify(reste) == 0


def test_q3_le_courant_initial_est_positif_vers_le_bas() -> None:
    """`i(0) = E0 racine(2) (R0+R1)/((R0+R1)^2 + (wL)^2)` : toujours positif."""
    somme = ov.R0 + ov.R1
    attendu = ov.E0 * sp.sqrt(2) * somme / (somme**2 + (ov.W * ov.L) ** 2)
    assert sp.simplify(ov.courant_initial() - attendu) == 0
    assert ov.courant_initial().is_positive


def test_q3_la_tension_est_negative() -> None:
    """Le courant de la bobine remonte dans `R2` : `u`, flechee vers le haut, < 0."""
    attendue = -ov.R2 * ov.courant_initial() * sp.exp(-(ov.R1 + ov.R2) * ov.T / ov.L)
    assert sp.simplify(ov.tension() - attendue) == 0


def test_q3_le_manuscrit_donne_l_oppose() -> None:
    """Page 3 : meme module, meme constante de temps, signe contraire."""
    assert sp.simplify(ov.tension() + ov.tension_du_manuscrit()) == 0


def test_q3_par_le_reseau_a_l_instant_du_basculement() -> None:
    """Second chemin : la bobine remplacee par une source du courant qu'elle porte."""
    assert (
        sp.simplify(ov.tension_initiale_par_le_reseau() - ov.tension().subs(ov.T, 0))
        == 0
    )


def test_q3_l_equation_d_apres_est_verifiee() -> None:
    """`(R1 + R2) i + L di/dt = 0`, avec `i` compte vers le bas dans la bobine."""
    apres = ov.courant_apres()
    assert sp.simplify((ov.R1 + ov.R2) * apres + ov.L * apres.diff(ov.T)) == 0


def test_q3_un_exemple_chiffre() -> None:
    """Avec toutes les valeurs a l'unite : `u(0+) = -racine(2) x 2/5`."""
    unites = dict.fromkeys((ov.E0, ov.W, ov.R0, ov.R1, ov.R2, ov.L), 1)
    depart = float(ov.tension().subs(unites).subs(ov.T, 0))
    assert depart == pytest.approx(-math.sqrt(2) * 2 / 5, rel=1e-12, abs=0)
