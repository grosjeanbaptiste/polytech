"""Electromagnetisme : l'exercice cote du 18 mai 2026 (Mons), trois questions.

Source : `Electromagnetisme/EXS/Exs_Cote_2025_2026_Mons_Solution.pdf`, cinq
pages numerotees 1, 2, 3, 5 et 6 -- enonce imprime, solutions
manuscrites --, transcrit dans
`transcriptions/Electromagnetisme/EXS/Exs_Cote_2025_2026_Mons_Solution.md`.

Chaque resultat du manuscrit est recopie comme oracle et recoupe par un
autre chemin : le courant de court-circuit, le couple sur un moment
magnetique, l'equation differentielle du circuit.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.application import cote_2026_mons_ouverture as ou
from polytech.electromagnetisme.application import cote_2026_mons_spires as sr
from polytech.electromagnetisme.application import cote_2026_mons_thevenin as th

# --- Question 1 : Thevenin en regime ----------------------------------------


def test_q1_le_generateur_equivalent() -> None:
    """Pages 1 et 2 : `R_T = 9/11` ohm (0,82), `E_T = 195/22` V (8,9)."""
    tension, resistance = th.thevenin()
    assert resistance == sp.Rational(9, 11)
    assert tension == sp.Rational(195, 22)
    assert float(resistance) == pytest.approx(0.82, abs=5e-3)
    assert float(tension) == pytest.approx(8.9, abs=5e-2)


def test_q1_les_courants_de_maille() -> None:
    """Page 2 : `I1 = 25/22` A ; la seconde maille donne alors `I2 = 5/22` A."""
    assert th.courants_de_maille() == (sp.Rational(25, 22), sp.Rational(5, 22))


def test_q1_la_tension_par_la_branche_de_gauche() -> None:
    """Page 2 : `E_T = E1 - R1 I1`."""
    premier, _ = th.courants_de_maille()
    assert th.E1 - th.R1 * premier == th.thevenin()[0]


def test_q1_le_court_circuit() -> None:
    """Second chemin : `A` relie a `B`, il passe `E_T / R_T = 65/6` A."""
    tension, resistance = th.thevenin()
    assert th.court_circuit() == sp.Rational(65, 6)
    assert tension / resistance == th.court_circuit()


def test_q1_ni_r4_ni_r5_ne_comptent() -> None:
    """Le condensateur ouvre `R4`, la bobine court-circuite `R5`."""
    assert th.thevenin(r4=40, r5=50) == th.thevenin()


# --- Question 2 : une petite spire au centre d'une grande -------------------


def test_q2_l_induction_au_centre() -> None:
    """Page 3 : `B(O) = -mu0 I/(2a) u_z`, le courant tournant en sens horaire."""
    assert sr.induction_au_centre() == (0, 0, -sr.MU0 * sr.I / (2 * sr.A))


def test_q2_l_induction_par_biot_et_savart() -> None:
    """Second chemin : l'integrale sur le cercle, parcouru en sens horaire."""
    x, y, z = sr.integrale_sur_le_cercle()
    assert (x, y) == pytest.approx((0, 0), abs=1e-12)
    assert z == pytest.approx(-2 * math.pi)


def test_q2_la_mutuelle_du_manuscrit() -> None:
    """Page 3 : `L_M = mu0 cos(alpha) dS/(2a)`."""
    attendue = sr.MU0 * sp.cos(sr.ALPHA) * sr.DS / (2 * sr.A)
    assert sp.simplify(sr.mutuelle() - attendue) == 0


def test_q2_le_couple_du_manuscrit() -> None:
    """Page 3 : `C = I I' dL_M/d(alpha) = -I I' mu0 dS sin(alpha)/(2a)`."""
    attendu = -sr.I * sr.IP * sr.MU0 * sr.DS * sp.sin(sr.ALPHA) / (2 * sr.A)
    assert sp.simplify(sr.couple() - attendu) == 0


def test_q2_le_couple_par_le_moment_magnetique() -> None:
    """Second chemin : `m x B`, de module `I' dS B sin(alpha)`."""
    x, y, z = sr.couple_sur_le_moment()
    assert (x, z) == (0, 0)
    assert sp.simplify(y + sr.couple()) == 0


def test_q2_le_couple_referme_l_angle() -> None:
    """Tourner la normale autour du couple fait decroitre `alpha`."""
    assert sr.pente_de_l_angle_sous_le_couple() == -1


def test_q2_le_couple_s_annule_quand_les_spires_sont_paralleles() -> None:
    """`alpha = 0` : le flux est maximal, la position est d'equilibre."""
    assert sr.couple().subs(sr.ALPHA, 0) == 0


# --- Question 3 : l'ouverture d'un interrupteur -----------------------------


def test_q3_juste_avant() -> None:
    """Page 5 : `I = -E1/(R1 + R)`, `Uc = R I = -R E1/(R1 + R)`."""
    somme = ou.R1 + ou.R
    assert ou.avant() == (-ou.E1 / somme, -ou.R * ou.E1 / somme)


def test_q3_juste_avant_par_le_reseau() -> None:
    """Second chemin : le circuit ferme, resolu comme un reseau en regime."""
    courant, tension = ou.avant_par_le_reseau()
    assert sp.simplify(courant - ou.avant()[0]) == 0
    assert sp.simplify(tension - ou.avant()[1]) == 0


def test_q3_la_transformee_du_manuscrit() -> None:
    """Page 6 : `I(p) = -E1/(R1 + R) (p L C + R C)/(L C p^2 + R C p + 1)`."""
    p = ou.P
    quotient = (p * ou.L * ou.C + ou.R * ou.C) / (
        ou.L * ou.C * p**2 + ou.R * ou.C * p + 1
    )
    attendue = -ou.E1 / (ou.R1 + ou.R) * quotient
    assert sp.simplify(ou.transformee() - attendue) == 0


def test_q3_un_quotient_de_deux_polynomes() -> None:
    """Le numerateur est du premier degre en `p`, le denominateur du second."""
    numerateur, denominateur = ou.polynomes()
    assert sp.degree(numerateur, ou.P) == 1
    assert sp.degree(denominateur, ou.P) == 2
    assert sp.simplify(numerateur / denominateur - ou.transformee()) == 0


def test_q3_par_l_equation_differentielle() -> None:
    """Second chemin : `LC i'' + RC i' + i = 0`, transformee avec `i'(0) = 0`."""
    assert ou.pente_initiale() == 0
    assert sp.simplify(ou.transformee_par_l_equation() - ou.transformee()) == 0


def test_q3_les_deux_limites() -> None:
    """La valeur initiale rend `i(0)`, la valeur finale rend zero."""
    initiale, finale = ou.limites()
    assert sp.simplify(initiale - ou.avant()[0]) == 0
    assert finale == 0
