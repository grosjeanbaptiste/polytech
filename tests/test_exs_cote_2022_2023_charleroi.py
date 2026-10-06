"""Electromagnetisme : l'exercice cote du 17 avril 2023 (Charleroi), trois questions.

Source : `Electromagnetisme/EXS/Exs_Cote_2022_2023_Charleroi_Solution.pdf`,
six pages -- enonce imprime, solutions manuscrites --, transcrit dans
`transcriptions/Electromagnetisme/EXS/Exs_Cote_2022_2023_Charleroi_Solution.md`.

Chaque resultat du manuscrit est recopie comme oracle et recoupe par un
autre chemin : le court-circuit et le diviseur de courant, l'inductance
mutuelle, la matrice reconstruite depuis les quatre equations.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.application import cote_2023_condensateurs as cd
from polytech.electromagnetisme.application import cote_2023_demi_tores as dt
from polytech.electromagnetisme.application import cote_2023_thevenin as th
from polytech.electromagnetisme.domain.magnetostatique.induction import (
    flux_a_entrefers_dominants,
)

# --- Le domaine --------------------------------------------------------------


def test_le_flux_quand_l_air_consomme_tous_les_amperes_tours() -> None:
    """`N I = H_e l` et `B = mu_0 H_e` : `Phi = mu_0 N I S / l`."""
    spires, courant, section, air = sp.symbols("N I S l", positive=True)
    flux = flux_a_entrefers_dominants(spires, courant, section, air)
    assert sp.simplify(flux - dt.MU0 * spires * courant * section / air) == 0


def test_un_circuit_sans_entrefer_n_a_pas_ce_flux() -> None:
    with pytest.raises(ValueError, match="entrefer"):
        flux_a_entrefers_dominants(10, 1, 1, sp.Integer(0))


# --- Question 1 : Thevenin autour d'une source de courant -------------------


def test_q1_le_generateur_equivalent() -> None:
    """Pages 1 et 2 : `R_T = 1,09` ohm, `E_T = 1,82` V -- soit 12/11 et 20/11."""
    tension, resistance = th.thevenin()
    assert resistance == sp.Rational(12, 11)
    assert tension == sp.Rational(20, 11)
    assert float(resistance) == pytest.approx(1.09, abs=5e-3)
    assert float(tension) == pytest.approx(1.82, abs=5e-3)


def test_q1_le_courant_du_manuscrit() -> None:
    """Page 2 : `I1 = R1 I / (R1 + R2 + R3 R4/(R3 + R4)) = 1,06` A."""
    assert th.courant_dans_r2() == sp.Rational(35, 33)
    assert float(th.courant_dans_r2()) == pytest.approx(1.06, abs=5e-3)


def test_q1_le_court_circuit() -> None:
    """`I_cc = E_T / R_T = 5/3` A : le diviseur de courant entre `R1` et `R2`."""
    assert th.courant_de_court_circuit() == sp.Rational(5, 3)
    tension, resistance = th.thevenin()
    assert tension / resistance == sp.Rational(5, 3)


def test_q1_le_condensateur_ne_compte_pas_en_regime() -> None:
    """Il est en parallele sur `R2` : ouvert, il laisse `R2` seule."""
    assert th.thevenin(avec_le_condensateur=False) == th.thevenin()


# --- Question 2 : un demi-tore qui s'eloigne -------------------------------


def test_q2_le_flux_et_la_force_electromotrice_du_manuscrit() -> None:
    """Pages 3 et 4 : le flux en `1/(le + vt)`, la fem en `1/(le + vt)^2`."""
    ecart = dt.ENTREFER + dt.VITESSE * dt.T
    flux, fem = dt.flux(), dt.fem()
    assert sp.simplify(flux - dt.MU0 * dt.N * dt.COURANT * dt.S / (2 * ecart)) == 0
    attendue = dt.MU0 * dt.VITESSE * dt.N**2 * dt.COURANT * dt.S / (2 * ecart**2)
    assert sp.simplify(fem - attendue) == 0


def test_q2_la_force_electromotrice_n_est_pas_constante() -> None:
    """Elle depend du temps, et decroit : le flux ne decroit pas lineairement."""
    fem = dt.fem()
    assert fem.has(dt.T)
    assert (
        sp.simplify(fem.diff(dt.T)).subs(
            {
                dt.MU0: 1,
                dt.N: 1,
                dt.COURANT: 1,
                dt.S: 1,
                dt.ENTREFER: 1,
                dt.VITESSE: 1,
                dt.T: 1,
            }
        )
        < 0
    )


def test_q2_par_l_inductance_mutuelle() -> None:
    """Second chemin : `M = mu0 N^2 S/(2x)`, puis `E = -I dM/dt`."""
    mutuelle = dt.inductance_mutuelle()
    assert sp.simplify(-dt.COURANT * mutuelle.diff(dt.T) - dt.fem()) == 0


def test_q2_l_aire_sous_la_fem_est_le_flux_initial() -> None:
    """`int E dt` de 0 a l'infini vaut `N Phi(0)` : tout le flux finit par partir."""
    assert sp.simplify(dt.aire_sous_la_fem() - dt.N * dt.flux().subs(dt.T, 0)) == 0


# --- Question 3 : une bobine basculee sur deux condensateurs ----------------


def test_q3_la_matrice_du_manuscrit() -> None:
    """Page 6 : quatre equations, inconnues `U_c1`, `U_c2`, `I1`, `I2`."""
    p = cd.P
    matrice, second_membre = cd.systeme()
    attendue = sp.Matrix(
        [
            [1, 0, p * cd.L, 0],
            [1, -1, 0, -cd.R2],
            [p * cd.C1, 0, -1, 1],
            [0, p * cd.C2, 0, -1],
        ]
    )
    assert matrice == attendue
    assert second_membre == sp.Matrix([cd.L * cd.I10, 0, 0, 0])


def test_q3_le_courant_initial_du_manuscrit() -> None:
    """Page 6 : `-E racine(2)/racine(R1^2 + w^2 L^2) cos(theta - arctan(wL/R1))`."""
    valeurs = {cd.E: 3, cd.W: 2, cd.R1: 5, cd.L: sp.Rational(7, 4), cd.THETA: 1}
    attendu = (-cd.E * sp.sqrt(2) / sp.sqrt(cd.R1**2 + (cd.W * cd.L) ** 2)) * sp.cos(
        cd.THETA - sp.atan(cd.W * cd.L / cd.R1)
    )
    ecart = (cd.courant_initial() - attendu).subs(valeurs)
    assert abs(sp.N(ecart, 30)) < sp.Float("1e-25")


def test_q3_le_courant_d_avant_verifie_l_equation_du_circuit() -> None:
    """Second chemin, sans phaseur : `e = -(R1 i1 + L di1/dt)`, au sens de `i1`."""
    avant = cd.courant_avant()
    source = cd.E * sp.sqrt(2) * sp.cos(cd.W * cd.T + cd.THETA)
    assert sp.simplify(source + cd.R1 * avant + cd.L * avant.diff(cd.T)) == 0


def test_q3_le_regime_final() -> None:
    """Tout s'eteint : la bobine court-circuite `C1`, puis `C2` a travers `R2`."""
    assert cd.regime() == (0, 0, 0, 0)


def test_q3_l_energie_de_la_bobine_se_dissipe_dans_r2() -> None:
    """`int R2 i2^2 dt = L i1(0)^2 / 2`, verifie sur un exemple chiffre."""
    dissipee, initiale = cd.bilan_d_energie_chiffre()
    assert dissipee == pytest.approx(initiale, rel=1e-9, abs=0)
