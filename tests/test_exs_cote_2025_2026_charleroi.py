"""Electromagnetisme : l'exercice cote du 7 mai 2026 (Charleroi), trois questions.

Source : `Electromagnetisme/EXS/Exs_Cote_2025_2026_Charleroi_Solution.pdf`,
six pages -- enonce imprime, solutions manuscrites --, transcrit dans
`transcriptions/Electromagnetisme/EXS/Exs_Cote_2025_2026_Charleroi_Solution.md`.

Chaque resultat du manuscrit est recopie comme oracle et recoupe par un
autre chemin : le theoreme de Millman, l'integrale de Biot et Savart,
l'equation differentielle du secondaire.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.application import cote_2026_couplage as cg
from polytech.electromagnetisme.application import cote_2026_spire as sr
from polytech.electromagnetisme.application import cote_2026_thevenin as th

# --- Question 1 : Thevenin a deux piles --------------------------------------


def test_q1_le_generateur_equivalent() -> None:
    """Pages 1 et 2 : `R_TH = 3/5 = 0,6` ohm et `E_TH = 7,5` V."""
    tension, resistance = th.thevenin()
    assert resistance == sp.Rational(3, 5)
    assert tension == sp.Rational(15, 2)


def test_q1_les_courants_du_manuscrit() -> None:
    """Page 2 : `I1 = 2,5` A ; la maille II donne alors `I2 = 1,25` A."""
    premier, second = th.courants_de_maille()
    assert premier == sp.Rational(5, 2)
    assert second == sp.Rational(5, 4)


def test_q1_par_le_theoreme_de_millman() -> None:
    """Second chemin : trois branches en parallele entre A et B."""
    tension, _ = th.thevenin()
    assert th.tension_par_millman() == tension


def test_q1_sans_la_bobine_le_resultat_change() -> None:
    """La bobine court-circuite `R3 + R4` : l'oublier donne d'autres valeurs."""
    assert th.thevenin(avec_la_bobine=False) != th.thevenin()


def test_q1_la_branche_capacitive_ne_compte_pas() -> None:
    assert th.thevenin(avec_la_branche_capacitive=False) == th.thevenin()


# --- Question 2 : deux fils et une petite spire ------------------------------


def test_q2_le_champ_du_manuscrit() -> None:
    """Page 4 : `H = I a/(pi (a^2 + R^2))` selon `x`, rien selon `y`."""
    h_x, h_y = sr.champ()
    assert sp.simplify(h_x - sr.I * sr.A / (sp.pi * (sr.A**2 + sr.R**2))) == 0
    assert h_y == 0


def test_q2_la_mutuelle_du_manuscrit() -> None:
    """Page 4 : `L_M = mu0 a dS/(pi (a^2 + R^2))`."""
    attendue = sr.MU0 * sr.A * sr.DS / (sp.pi * (sr.A**2 + sr.R**2))
    assert sp.simplify(sr.mutuelle() - attendue) == 0


def test_q2_la_force_du_manuscrit() -> None:
    """Page 4 : `F_x = -2 mu0 a I I' dS R/(pi (a^2 + R^2)^2)`."""
    carre = (sr.A**2 + sr.R**2) ** 2
    attendue = -2 * sr.MU0 * sr.A * sr.I * sr.IP * sr.DS * sr.R / (sp.pi * carre)
    assert sp.simplify(sr.force() - attendue) == 0


def test_q2_la_force_ramene_la_spire_vers_les_fils() -> None:
    """Negative pour `R > 0` : la spire est attiree vers le champ fort."""
    assert sr.force().is_negative


def test_q2_la_force_est_la_plus_grande_en_a_sur_racine_de_trois() -> None:
    assert sr.distance_de_la_force_maximale() == sr.A / sp.sqrt(3)


@pytest.mark.parametrize(("demi_ecart", "distance"), [(1.0, 0.5), (0.3, 2.0)])
def test_q2_le_champ_par_biot_et_savart(demi_ecart: float, distance: float) -> None:
    """Second chemin : les deux fils integres element par element."""
    valeurs = {sr.I: 1, sr.A: demi_ecart, sr.R: distance}
    h_x, _ = sr.champ()
    par_x, par_y = sr.champ_par_biot_et_savart(demi_ecart, distance)
    assert par_x == pytest.approx(float(h_x.subs(valeurs)), rel=1e-6, abs=0)
    assert par_y == pytest.approx(0.0, abs=1e-9)


def test_q2_la_force_par_le_gradient_du_champ_integre() -> None:
    """Second chemin : `F = I' dS mu0 dH_x/dR`, la derivee prise numeriquement."""
    pas = 1e-4
    plus, _ = sr.champ_par_biot_et_savart(1.0, 0.5 + pas)
    moins, _ = sr.champ_par_biot_et_savart(1.0, 0.5 - pas)
    unites = {sr.I: 1, sr.IP: 1, sr.DS: 1, sr.MU0: 1, sr.A: 1, sr.R: 0.5}
    attendue = float(sr.force().subs(unites))
    assert (plus - moins) / (2 * pas) == pytest.approx(attendue, rel=1e-5, abs=0)


# --- Question 3 : deux bobines couplees en quadrature ------------------------


def test_q3_les_points_donnent_une_mutuelle_negative() -> None:
    """`i1` entre par la borne sans point, `i2` par la borne pointee."""
    assert cg.signe() == -1


def test_q3_le_rapport_du_manuscrit() -> None:
    """Page 5 : `I2/I1 = j w L12/(R2 + j w L2 + 1/(j w C))`."""
    attendu = (
        sp.I * cg.W * cg.L12 / (cg.R2 + sp.I * cg.W * cg.L2 + 1 / (sp.I * cg.W * cg.C))
    )
    assert sp.simplify(cg.rapport() - attendu) == 0


def test_q3_la_condition_du_manuscrit() -> None:
    """Page 6 : `L2 C = 1/w^2`, soit `1/(10^4 pi^2)` a 50 Hz."""
    assert cg.condition() == sp.Eq(cg.L2 * cg.C, 1 / cg.W**2)
    assert cg.produit_a_cinquante_hertz() == 1 / (10**4 * sp.pi**2)


def test_q3_sous_la_condition_le_rapport_est_imaginaire_positif() -> None:
    """`I2/I1 = j w L12/R2` : `i2` en avance d'un quart de periode."""
    assert sp.simplify(cg.rapport_a_la_resonance() - sp.I * cg.W * cg.L12 / cg.R2) == 0


def test_q3_la_mutuelle_du_manuscrit() -> None:
    """Page 6 : `L12 = 1/(20 pi)` H, que le manuscrit arrondit a 16 mH."""
    assert cg.mutuelle() == 1 / (20 * sp.pi)
    assert float(cg.mutuelle()) * 1e3 == pytest.approx(15.9, abs=0.05)
    assert round(float(cg.mutuelle()) * 1e3) == 16


def test_q3_par_l_equation_differentielle_du_secondaire() -> None:
    """Second chemin, sans phaseur : `i2 = (w L12/R2) cos(w t + pi/2)` la verifie."""
    assert sp.simplify(cg.residu_de_l_equation_du_secondaire()) == 0


def test_q3_des_points_inverses_donneraient_un_retard() -> None:
    """Le signe de la mutuelle decide du sens du quart de periode."""
    inverse = cg.rapport_a_la_resonance(signe=1)
    assert sp.simplify(inverse + sp.I * cg.W * cg.L12 / cg.R2) == 0


def test_q3_la_resistance_ramenee_au_primaire() -> None:
    """`(w L12)^2/R2 = 2,5` ohms : le secondaire accorde charge le primaire."""
    assert cg.resistance_ramenee() == sp.Rational(5, 2)
    assert math.isclose(float(cg.resistance_ramenee()), 2.5)
