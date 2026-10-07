"""Electromagnetisme : l'exercice cote du 19 mai 2025 (Mons), trois questions.

Sources : l'enonce, `Electromagnetisme/EXS/Exs_Cote_2024_2025_Mons_Enonce.pdf`
(sept pages imprimees), et sa solution manuscrite,
`Exs_Cote_2024_2025_Mons_Solution.pdf` (trois pages), transcrite dans
`transcriptions/Electromagnetisme/EXS/Exs_Cote_2024_2025_Mons_Solution.md`.

Chaque resultat du manuscrit est recopie comme oracle et recoupe par un
autre chemin : le reseau resolu avec son dipole, la pression de Maxwell
dans les entrefers, le rapport des courants chiffre.
"""

from __future__ import annotations

import cmath
import math

import pytest
import sympy as sp

from polytech.electromagnetisme.application import cote_2025_mons_electroaimant as el
from polytech.electromagnetisme.application import cote_2025_mons_pont as po
from polytech.electromagnetisme.application import cote_2025_mons_thevenin as th

# --- Question 1 : Thevenin, puis une pile et une resistance -----------------


def test_q1_le_generateur_equivalent() -> None:
    """Page 1 : `R_TH = 4/3` ohm, `E_TH = -4` V."""
    tension, resistance = th.thevenin()
    assert resistance == sp.Rational(4, 3)
    assert tension == -4


def test_q1_le_courant_de_r2() -> None:
    """Page 1 : `I1 = (-E2 - R4 I)/(R2 + R4) = -3` A, compte vers le haut."""
    assert th.courant_dans_r2() == -3


def test_q1_par_millman() -> None:
    """Second chemin : `(I - E2/R2)/(1/R2 + 1/R4)`."""
    assert th.tension_par_millman() == -4


def test_q1_la_branche_de_droite_ne_compte_pas() -> None:
    """Le condensateur ouvre la branche `R1`, `E1`, `C`, `R3`."""
    assert th.thevenin(e1=150, r3=30) == th.thevenin()


def test_q1_la_tension_du_manuscrit() -> None:
    """Page 1 : `U = -R'/(R_TH + R') (E_TH - E') = +5` V."""
    assert th.tension_sur_la_resistance() == 5


def test_q1_le_dipole_sur_le_circuit() -> None:
    """Second chemin : le dipole branche sur le circuit d'origine."""
    assert th.tension_par_le_circuit() == 5


# --- Question 2 : un electro-aimant et sa plaque ----------------------------


def test_q2_le_flux_du_manuscrit() -> None:
    """Page 2 : `B_H = 0,3` T lu pour 4 A/m, donc `Phi = 0,00012` Wb."""
    assert el.flux() == pytest.approx(0.00012)


def test_q2_l_induction_dans_la_plaque() -> None:
    """Page 2 : meme flux, meme section : `B_TO = 0,3` T, et `H_TO = 55` A/m lu."""
    assert el.induction_dans_la_plaque() == pytest.approx(0.3)
    assert el.CHAMP_LU_DANS_LES_TOLES == 55.0


def test_q2_le_courant_du_manuscrit() -> None:
    """Page 2 : `55 x 0,16 + 4 pi 0,07 + 2 (0,3/mu0) 0,002 = 200 I`, `I = 4,8` A."""
    assert el.courant() == pytest.approx(4.8, abs=5e-2)
    assert el.courant() == pytest.approx(4.823, abs=5e-4)


def test_q2_le_partage_des_amperes_tours() -> None:
    """Hipersil 0,88 ; toles 8,8 ; entrefers 954,9 : l'air en prend 99 %."""
    hipersil, toles, air = el.partage()
    assert hipersil == pytest.approx(0.88, abs=5e-3)
    assert toles == pytest.approx(8.8)
    assert air == pytest.approx(954.9, abs=5e-2)
    assert air / (hipersil + toles + air) == pytest.approx(0.99, abs=5e-4)


def test_q2_la_force_du_manuscrit() -> None:
    """Page 2 : `|F| = (1/4) I^2 N^2 mu0 S / le^2 = 29,0` N avec `I = 4,8` A."""
    assert el.force(4.8) == pytest.approx(29.0, abs=5e-2)


def test_q2_la_force_par_la_pression_de_maxwell() -> None:
    """Second chemin : `B^2/(2 mu0)` sur chacun des deux poles."""
    assert el.force_par_la_pression(4.8) == pytest.approx(el.force(4.8))


def test_q2_l_induction_de_l_approximation() -> None:
    """L'air seul : `mu0 N I/(2 le) = 0,302` T, pour les 0,3 T du calcul complet."""
    assert el.induction_approchee(4.8) == pytest.approx(0.302, abs=5e-4)


def test_q2_la_plaque_se_souleve() -> None:
    """Page 2 : `P = 0,5 x 9,81 = 4,905` N, inferieur a la force."""
    assert el.poids() == pytest.approx(4.905)
    assert el.force(4.8) > el.poids()
    assert el.force(el.courant()) > el.poids()


def test_q2_le_courant_qui_suffirait() -> None:
    """La force va comme `I^2` : 1,98 A equilibrent deja le poids."""
    assert el.courant_minimal() == pytest.approx(1.98, abs=5e-3)
    assert el.force(el.courant_minimal()) == pytest.approx(el.poids())


# --- Question 3 : deux branches en parallele --------------------------------


def test_q3_la_matrice_du_manuscrit() -> None:
    """Page 3 : deux mailles, inconnues `I1` et `I2`."""
    serie = po.R1 + sp.I * po.W * po.L
    attendue = sp.Matrix(
        [
            [po.R0 + serie + 1 / (sp.I * po.W * po.C), -serie],
            [-serie, po.R2 + serie],
        ]
    )
    matrice, second_membre = po.systeme()
    assert sp.simplify(matrice - attendue) == sp.zeros(2, 2)
    assert second_membre == sp.Matrix([po.E0, 0])


def test_q3_le_rapport_du_manuscrit() -> None:
    """Page 3 : `I1/I2 = (R1 + R2 + j w L)/(R1 + j w L)`."""
    serie = po.R1 + sp.I * po.W * po.L
    assert sp.simplify(po.rapport() - (serie + po.R2) / serie) == 0


def test_q3_le_dephasage_du_manuscrit() -> None:
    """Page 3 : `atan(w L/(R1 + R2)) - atan(w L/R1)`."""
    attendu = sp.atan(po.W * po.L / (po.R1 + po.R2)) - sp.atan(po.W * po.L / po.R1)
    assert sp.simplify(po.dephasage() - attendu) == 0


def test_q3_le_dephasage_sur_un_exemple() -> None:
    """Second chemin : l'argument du rapport, pour des valeurs quelconques."""
    valeurs = {po.R1: 3, po.R2: 5, po.L: 2, po.W: 7}
    rapport = complex(po.rapport().subs(valeurs))
    assert float(po.dephasage().subs(valeurs)) == pytest.approx(cmath.phase(rapport))
    assert cmath.phase(rapport) < 0


def test_q3_l_impedance_a_ajouter() -> None:
    """Page 3 : `Z = (R1 - R2) + j w L`."""
    attendue = po.R1 - po.R2 + sp.I * po.W * po.L
    assert sp.simplify(po.impedance_a_ajouter() - attendue) == 0


def test_q3_avec_elle_le_courant_se_partage_en_deux() -> None:
    """Second chemin : `Z` mise en serie avec `R2`, le rapport vaut deux."""
    assert po.rapport(en_serie=po.impedance_a_ajouter()) == 2


def test_q3_la_condition() -> None:
    """Page 3 : possible si `R1 > R2` ; a l'egalite, `Z` est une bobine pure."""
    assert po.est_realisable(r1=3, r2=2)
    assert po.est_realisable(r1=2, r2=2)
    assert not po.est_realisable(r1=2, r2=3)
    assert math.isclose(
        float(sp.re(po.impedance_a_ajouter().subs({po.R1: 2, po.R2: 2}))), 0
    )
