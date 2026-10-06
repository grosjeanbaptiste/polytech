"""Electromagnetisme : l'exercice cote du 29 avril 2022 (Charleroi), trois questions.

Source : `Electromagnetisme/EXS/Exs_Cote_2021_2022_Charleroi_Solution.pdf`,
six pages -- enonce imprime, solutions manuscrites --, transcrit dans
`transcriptions/Electromagnetisme/EXS/Exs_Cote_2021_2022_Charleroi_Solution.md`.

Chaque resultat du manuscrit est recopie comme oracle et recoupe par un
autre chemin : la divergence du champ trouve, Millman et le court-circuit,
l'equation du circuit dans le temps.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.application import cote_2022_bascule as ba
from polytech.electromagnetisme.application import cote_2022_coquille as cq
from polytech.electromagnetisme.application import cote_2022_thevenin as th

R, R1, R2 = cq.RAYON, cq.R1, cq.R2

# --- Question 1 : une coquille chargee, puis une coquille conductrice ------


def test_q1_les_trois_champs_du_manuscrit() -> None:
    """Pages 1 et 2 : nul dans la cavite, puis les deux expressions du manuscrit."""
    cavite, coquille, dehors = cq.champs()
    assert cavite == 0
    assert sp.simplify(coquille - cq.ETA / (3 * cq.EPS) * (R - R1**3 / R**2)) == 0
    attendu = cq.ETA * (R2**3 - R1**3) / (3 * cq.EPS0 * R**2)
    assert sp.simplify(dehors - attendu) == 0


def test_q1_la_divergence_redonne_la_charge() -> None:
    """Second chemin : `div(eps E) = eta` dans la coquille, zero dehors."""
    dans_la_coquille, au_dehors = cq.divergences()
    assert sp.simplify(dans_la_coquille - cq.ETA) == 0
    assert sp.simplify(au_dehors) == 0


def test_q1_le_champ_part_de_zero_et_saute_a_la_sortie() -> None:
    """Continu en `R1` ; en `R2` c'est `D` qui est continu, `E` saute de `eps/eps0`."""
    _, coquille, dehors = cq.champs()
    assert sp.simplify(coquille.subs(R, R1)) == 0
    rapport = sp.simplify(dehors.subs(R, R2) / coquille.subs(R, R2))
    assert rapport == cq.EPS / cq.EPS0


def test_q1_la_charge_de_la_face_interieure() -> None:
    """Page 2 : `sigma = -q / (4 pi R1^2)`."""
    interieure, exterieure = cq.charges_surfaciques()
    assert sp.simplify(interieure + cq.Q / (4 * sp.pi * R1**2)) == 0
    assert sp.simplify(exterieure - cq.Q / (4 * sp.pi * R2**2)) == 0
    totale = interieure * 4 * sp.pi * R1**2 + exterieure * 4 * sp.pi * R2**2
    assert sp.simplify(totale) == 0


# --- Question 2 : Thevenin, puis l'energie d'un condensateur ---------------


def test_q2_le_generateur_equivalent() -> None:
    """Pages 3 et 4 : `R_TH = 2/3` ohm, `E_TH = 4` V."""
    tension, resistance = th.thevenin()
    assert resistance == sp.Rational(2, 3)
    assert tension == 4


def test_q2_le_courant_de_la_maille_du_manuscrit() -> None:
    """Page 4 : `I4 = (E + R1 I)/(R1 + R3 + R4) = 4` A."""
    assert th.courant_dans_r4() == 4


def test_q2_le_court_circuit() -> None:
    """`I_cc = E_TH / R_TH = 6` A, mesure sur le reseau complet."""
    assert th.courant_de_court_circuit() == 6


def test_q2_l_energie_du_condensateur() -> None:
    """Page 4 : `U0 = 60/17` V, `6,23` microjoules."""
    tension, energie = th.charge_du_dipole()
    assert tension == sp.Rational(60, 17)
    assert float(energie) * 1e6 == pytest.approx(6.23, abs=5e-3)


def test_q2_la_meme_tension_sur_le_circuit_entier() -> None:
    """Le dipole branche sur le reseau d'origine, sans equivalent."""
    assert th.tension_sur_le_reseau_complet() == th.charge_du_dipole()[0]


# --- Question 3 : basculer une bobine d'une source vers une resistance -----


def test_q3_le_courant_avant_le_basculement() -> None:
    """Page 6 : `E racine(2)/racine(R0^2 + (wL)^2)`, en retard de `arctan(wL/R0)`."""
    valeurs = {ba.E: 3, ba.W: 2, ba.R0: 5, ba.L: sp.Rational(7, 4), ba.THETA: 1}
    attendu = (ba.E * sp.sqrt(2) / sp.sqrt(ba.R0**2 + (ba.W * ba.L) ** 2)) * sp.cos(
        ba.W * ba.T + ba.THETA - sp.atan(ba.W * ba.L / ba.R0)
    )
    for instant in (0, sp.Rational(1, 3), 2):
        ecart = (ba.courant_avant() - attendu).subs(valeurs).subs(ba.T, instant)
        assert abs(sp.N(ecart, 30)) < sp.Float("1e-25")


def test_q3_ce_courant_verifie_l_equation_du_circuit() -> None:
    """Second chemin, sans phaseur : `R0 i + L di/dt = e(t)`."""
    avant = ba.courant_avant()
    source = ba.E * sp.sqrt(2) * sp.cos(ba.W * ba.T + ba.THETA)
    assert sp.simplify(ba.R0 * avant + ba.L * avant.diff(ba.T) - source) == 0


def test_q3_la_tension_de_la_bobine() -> None:
    """Page 6 : `u_L = -R i(0) e^(-R t / L)`."""
    depart = ba.courant_avant().subs(ba.T, 0)
    attendue = -ba.R * depart * sp.exp(-ba.R * ba.T / ba.L)
    assert sp.simplify(ba.tension_de_la_bobine() - attendue) == 0


def test_q3_la_tension_est_celle_que_la_bobine_impose() -> None:
    """`u_L = L di/dt` : la meme, par la loi de la bobine."""
    courant = ba.courant_apres()
    assert sp.simplify(ba.tension_de_la_bobine() - ba.L * courant.diff(ba.T)) == 0


def test_q3_la_tension_saute_au_basculement() -> None:
    """Avant : `e(0) - R0 i(0)`. Apres : `-R i(0)`. Le courant, lui, ne saute pas."""
    avant, apres = ba.tensions_au_basculement()
    depart = ba.courant_avant().subs(ba.T, 0)
    source = ba.E * sp.sqrt(2) * sp.cos(ba.THETA)
    assert sp.simplify(avant - (source - ba.R0 * depart)) == 0
    assert sp.simplify(apres + ba.R * depart) == 0
