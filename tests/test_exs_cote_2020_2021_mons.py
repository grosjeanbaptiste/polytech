"""Electromagnetisme : l'exercice cote du 17 mai 2021 (Mons), trois questions.

Source : `Electromagnetisme/EXS/Exs_Cote_2020_2021_Mons_Solution.pdf`, huit
pages -- enonce imprime, solutions manuscrites, la derniere blanche --,
transcrit dans
`transcriptions/Electromagnetisme/EXS/Exs_Cote_2020_2021_Mons_Solution.md`.

Chaque resultat du manuscrit est recopie comme oracle et recoupe par un
autre chemin : le bilan de puissance, le court-circuit, l'equation
differentielle du condensateur.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.application import cote_2021_mons_capacite as cp
from polytech.electromagnetisme.application import cote_2021_mons_rails as ra
from polytech.electromagnetisme.application import cote_2021_mons_thevenin as th

# --- Question 1 : une tige qui glisse sur deux rails ------------------------


def test_q1_la_force_electromotrice() -> None:
    """Page 1 : `B delta v = 1,5 x 0,5 x 30 = 22,5` V."""
    assert ra.fem() == pytest.approx(22.5, rel=1e-12)


def test_q1_par_la_derivee_du_flux() -> None:
    """Second chemin : `-d/dt [B delta (l0 - v t)]`, la spire retrecissant."""
    assert float(ra.fem_par_le_flux()) == pytest.approx(ra.fem(), rel=1e-12)


def test_q1_le_courant_et_la_puissance() -> None:
    """Page 2 : `I = 2,25` A, `P = R I^2 = 50,625` W."""
    assert ra.courant() == pytest.approx(2.25, rel=1e-12)
    assert ra.puissance() == pytest.approx(50.625, rel=1e-12)


def test_q1_la_force() -> None:
    """Page 2 : `B delta I = 1,6875` N, selon `+x` -- a l'oppose de la vitesse."""
    selon_x, selon_y, selon_z = ra.force()
    assert selon_x == pytest.approx(1.6875, rel=1e-12)
    assert (selon_y, selon_z) == (0.0, 0.0)
    assert selon_x * ra.VITESSE[0] < 0


def test_q1_la_puissance_mecanique_egale_la_puissance_electrique() -> None:
    """Second chemin : ce que l'on fournit pour pousser la tige part en chaleur."""
    assert ra.puissance_mecanique() == pytest.approx(ra.puissance(), rel=1e-12)


# --- Question 2 : Thevenin, puis une bobine resistive -----------------------


def test_q2_le_generateur_equivalent() -> None:
    """Page 3 : `R_TH = 1/3 + 1 = 4/3` ohm, `E_TH = 1` V."""
    tension, resistance = th.thevenin()
    assert resistance == sp.Rational(4, 3)
    assert tension == 1


def test_q2_le_courant_du_dipole() -> None:
    """Page 4 : `I' = E_TH/(R_TH + R0) = 3/19` A."""
    assert th.courant_par_l_equivalent() == sp.Rational(3, 19)


def test_q2_par_le_circuit_entier() -> None:
    """Second chemin : le dipole branche sur le circuit d'origine, sans equivalent."""
    assert th.courant_par_le_circuit() == sp.Rational(3, 19)


def test_q2_par_le_court_circuit() -> None:
    """`I_cc = E_TH/R_TH = 3/4` A, mesure sur le reseau."""
    tension, resistance = th.thevenin()
    assert th.courant_de_court_circuit() == tension / resistance


# --- Question 3 : charger seize microsecondes, puis decharger ---------------


def test_q3_la_charge() -> None:
    """Page 6 : `u_c = 10 (1 - e^(-t/10^-5))` pour `t <= 16` microsecondes."""
    t = cp.T
    attendue = 10 * (1 - sp.exp(-t / sp.Rational(1, 10**5)))
    assert sp.simplify(cp.charge() - attendue) == 0


def test_q3_la_tension_au_basculement() -> None:
    """Page 6 : `10 (1 - e^(-1,6)) = 7,98` V."""
    assert float(cp.tension_au_basculement()) == pytest.approx(7.98, abs=5e-3)


def test_q3_la_decharge() -> None:
    """Page 6 : `7,98 e^(-(t - 16.10^-6)/10^-5)`."""
    t = cp.T
    retard = t - sp.Rational(16, 10**6)
    attendue = cp.tension_au_basculement() * sp.exp(-retard / sp.Rational(1, 10**5))
    assert sp.simplify(cp.decharge() - attendue) == 0


@pytest.mark.parametrize(
    ("microsecondes", "volts"),
    [(8, 5.51), (16, 7.98), (24, 3.59), (32, 1.61)],
)
def test_q3_les_points_du_graphique(microsecondes: int, volts: float) -> None:
    """Page 7 : les ordonnees portees a 16 et 24 microsecondes, et les deux voisines."""
    assert cp.tension(microsecondes * 1e-6) == pytest.approx(volts, abs=6e-3)


def test_q3_la_tension_est_continue_au_basculement() -> None:
    """Un condensateur ne saute pas : la charge et la decharge se raccordent."""
    instant = cp.BASCULEMENT
    assert (
        sp.simplify(cp.charge().subs(cp.T, instant) - cp.decharge().subs(cp.T, instant))
        == 0
    )


def test_q3_les_deux_morceaux_verifient_leur_equation() -> None:
    """Second chemin, sans Laplace : `E0 = R0 C u' + u`, puis `0 = R C u' + u`."""
    charge, decharge = cp.charge(), cp.decharge()
    assert sp.simplify(cp.R0 * cp.C * charge.diff(cp.T) + charge - cp.E0) == 0
    assert sp.simplify(cp.R * cp.C * decharge.diff(cp.T) + decharge) == 0


def test_q3_l_energie_dissipee_a_la_decharge() -> None:
    """Ce que `R` dissipe apres le basculement est ce que le condensateur portait."""
    stockee = sp.Rational(1, 2) * cp.C * cp.tension_au_basculement() ** 2
    assert sp.simplify(cp.energie_dissipee_a_la_decharge() - stockee) == 0
    assert float(stockee) * 1e6 == pytest.approx(31.8, abs=0.05)
    assert math.isclose(float(cp.constante_de_temps()), 1e-5)
