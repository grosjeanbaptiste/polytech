"""Electromagnetisme : l'exercice cote du 23 mai 2022 (Mons), trois questions.

Source : `Electromagnetisme/EXS/Exs_Cote_2021_2022_Solution.pdf`, six pages
-- enonce imprime, solutions manuscrites ; le mot « Mons » est ecrit a la
main en tete --, transcrit dans
`transcriptions/Electromagnetisme/EXS/Exs_Cote_2021_2022_Solution.md`.

Chaque resultat du manuscrit est recopie comme oracle et recoupe par un
autre chemin : le flux integre sur le disque, le theoreme de Millman, les
trois mailles resolues en nombres.
"""

from __future__ import annotations

import cmath
import math

import pytest
import sympy as sp

from polytech.electromagnetisme.application import cote_2022_mons_couplage as cg
from polytech.electromagnetisme.application import cote_2022_mons_spire as sr
from polytech.electromagnetisme.application import cote_2022_mons_thevenin as th

# --- Question 1 : une spire qui tombe dans un champ non uniforme ------------


def test_q1_le_champ_est_bien_un_champ_magnetique() -> None:
    """`div B = -B + 0 + B = 0` : l'induction de l'enonce est admissible."""
    assert sr.divergence() == 0


def test_q1_le_flux_du_manuscrit() -> None:
    """Page 1 : `phi = B pi R^2 (h - g t^2/2)`."""
    attendu = sr.B * sp.pi * sr.R**2 * (sr.H - sr.G * sr.T**2 / 2)
    assert sp.simplify(sr.flux() - attendu) == 0


def test_q1_le_flux_par_l_integrale_sur_le_disque() -> None:
    """Second chemin : `B_z` integre sur le disque, en coordonnees polaires."""
    assert sp.simplify(sr.flux_par_integration() - sr.flux()) == 0


def test_q1_la_force_electromotrice_du_manuscrit() -> None:
    """Page 2 : `|E| = B pi R^2 g t`."""
    assert sp.simplify(sr.fem_absolue() - sr.B * sp.pi * sr.R**2 * sr.G * sr.T) == 0


def test_q1_la_duree_de_la_chute() -> None:
    """Page 2 : la spire touche le sol en `t = racine(2h/g)`, puis `|E| = 0`."""
    assert sr.duree_de_la_chute() == sp.sqrt(2 * sr.H / sr.G)


def test_q1_la_valeur_au_sol() -> None:
    """`B pi R^2 racine(2 g h)` : la force electromotrice est `B S` fois la vitesse."""
    au_sol = sr.fem_absolue().subs(sr.T, sr.duree_de_la_chute())
    attendue = sr.B * sp.pi * sr.R**2 * sp.sqrt(2 * sr.G * sr.H)
    assert sp.simplify(au_sol - attendue) == 0


# --- Question 2 : Thevenin, puis une bobine reelle --------------------------


def test_q2_le_generateur_equivalent() -> None:
    """Pages 3 et 4 : `R_TH = 1/3` ohm, `E_TH = 4` V."""
    tension, resistance = th.thevenin()
    assert resistance == sp.Rational(1, 3)
    assert tension == 4


def test_q2_par_millman() -> None:
    """Second chemin : `(I + E/R4)/(1/R1 + 1/R3 + 1/R4)`."""
    assert th.tension_par_millman() == 4


def test_q2_les_courants_du_manuscrit() -> None:
    """Page 4 : `I3 = 4` A, et `I4 = I3 - 10 = -6` A."""
    assert th.courants_de_maille() == (4, -6)


def test_q2_la_puissance_dans_la_bobine() -> None:
    """Page 4 : `I = 4 x 30/13 = 9,23` A, `P = 0,1 x 9,23^2 = 8,52` W."""
    courant, puissance = th.bobine()
    assert courant == sp.Rational(120, 13)
    assert float(courant) == pytest.approx(9.23, abs=5e-3)
    assert float(puissance) == pytest.approx(8.52, abs=5e-3)


def test_q2_par_le_circuit_entier() -> None:
    """Second chemin : la bobine branchee sur le circuit d'origine."""
    assert th.courant_par_le_circuit() == sp.Rational(120, 13)


# --- Question 3 : trois mailles, deux bobines couplees ----------------------


def test_q3_le_signe_de_la_mutuelle() -> None:
    """`i1` entre dans `L1` par la borne sans point, `i2` dans `L2` par le point."""
    assert cg.signe() == -1


def test_q3_la_matrice_du_manuscrit() -> None:
    """Page 6 : trois mailles, inconnues `I0`, `I1`, `I2`."""
    w = cg.W
    zc = 1 / (sp.I * w * cg.C)
    attendue = sp.Matrix(
        [
            [cg.R0 + cg.R1 + zc, -cg.R1 - zc, 0],
            [-cg.R1 - zc, cg.R1 + sp.I * w * cg.L1 + zc + cg.R2, -sp.I * w * cg.L12],
            [0, -sp.I * w * cg.L12, cg.R3 + sp.I * w * cg.L2],
        ]
    )
    matrice, second_membre = cg.systeme()
    assert sp.simplify(matrice - attendue) == sp.zeros(3, 3)
    assert second_membre == sp.Matrix([cg.E0, 0, 0])


def test_q3_la_matrice_est_symetrique() -> None:
    """Un reseau passif vu par ses mailles : sa matrice l'est toujours."""
    matrice, _ = cg.systeme()
    assert sp.simplify(matrice - matrice.T) == sp.zeros(3, 3)


def test_q3_le_rapport_du_manuscrit() -> None:
    """Page 6 : `I2/I1 = 0,0021 + 0,0100 j = 0,0102 e^(1,3643 j)`."""
    rapport = cg.rapport_chiffre()
    assert rapport.real == pytest.approx(0.0021, abs=5e-5)
    assert rapport.imag == pytest.approx(0.0100, abs=5e-5)
    assert abs(rapport) == pytest.approx(0.0102, abs=5e-5)
    assert cmath.phase(rapport) == pytest.approx(1.3643, abs=5e-5)


def test_q3_le_dephasage_du_manuscrit() -> None:
    """Page 6 : `78,17` degres, soit `90 - arctan(w L2/R3)`."""
    assert cg.dephasage_en_degres() == pytest.approx(78.17, abs=5e-3)
    attendu = 90 - math.degrees(math.atan(2 * math.pi * 50 * 2e-3 / 3))
    assert cg.dephasage_en_degres() == pytest.approx(attendu, abs=1e-9)


def test_q3_par_la_resolution_des_trois_mailles() -> None:
    """Second chemin : le systeme entier resolu, avec des valeurs pour ce qui manque."""
    premier, second = cg.courants_resolus(r0=5, r1=7, capacite=1e-4)
    assert cmath.phase(second / premier) == pytest.approx(
        math.radians(cg.dephasage_en_degres()), abs=1e-9
    )
