"""Electromagnetisme : l'exercice cote du 16 avril 2024 (Charleroi), trois questions.

Source : `Electromagnetisme/EXS/Exs_Cote_2023_2024_Charleroi_Solution.pdf`,
cinq pages -- enonce imprime, solutions manuscrites --, transcrit dans
`transcriptions/Electromagnetisme/EXS/Exs_Cote_2023_2024_Charleroi_Solution.md`.

Chaque resultat du manuscrit est recopie comme oracle et recoupe par un
autre chemin : le diviseur de tension, l'integrale de Biot et Savart, le
reseau complet resolu en complexes.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.application import cote_2024_fils as fl
from polytech.electromagnetisme.application import cote_2024_phase as ph
from polytech.electromagnetisme.application import cote_2024_thevenin as th

# --- Question 1 : Thevenin, puis un condensateur ----------------------------


def test_q1_le_generateur_equivalent() -> None:
    """Page 2 : `R_T = 1,5` ohm et `E_T = -5` V, de B vers A."""
    tension, resistance = th.thevenin()
    assert resistance == sp.Rational(3, 2)
    assert tension == -5


def test_q1_par_le_diviseur_de_tension() -> None:
    """Second chemin : `V_B - V_A` est la part de `E` que prennent `R1 + R2`."""
    tension, _ = th.thevenin()
    assert th.tension_par_le_diviseur() == -tension


def test_q1_le_diviseur_ne_doit_rien_a_l_egalite_des_resistances() -> None:
    """L'enonce a `R1 + R2 = R3` : d'autres valeurs separent les deux chemins."""
    autres = th.Resistances(sp.Integer(2), sp.Integer(5), sp.Integer(3), sp.Integer(4))
    tension, _ = th.thevenin(resistances=autres)
    assert tension == -7
    assert th.tension_par_le_diviseur(autres) == 7


def test_q1_l_energie_du_condensateur() -> None:
    """Page 2 : `1/2 C0 E_T^2 = 25` microjoules."""
    assert th.energie() == pytest.approx(25e-6, rel=1e-12, abs=0)


def test_q1_sans_la_bobine_le_resultat_change() -> None:
    """La bobine court-circuite `R4` : l'oublier donne `-7` V et `2,1` ohms."""
    tension, resistance = th.thevenin(avec_la_bobine=False)
    assert tension == -7
    assert resistance == sp.Rational(21, 10)


# --- Question 2 : la force sur un fil entre deux conducteurs ----------------


def test_q2_le_champ_du_manuscrit() -> None:
    """Page 4 : `H_x = I2 y/(2 pi (d^2 + y^2)) - I1/(2 pi y)`, `H_y = I2 d/(...)`."""
    h_x, h_y = fl.champ()
    carre = fl.D**2 + fl.Y**2
    attendu_x = fl.I2 * fl.Y / (2 * sp.pi * carre) - fl.I1 / (2 * sp.pi * fl.Y)
    assert sp.simplify(h_x - attendu_x) == 0
    assert sp.simplify(h_y - fl.I2 * fl.D / (2 * sp.pi * carre)) == 0


def test_q2_les_ordonnees_du_manuscrit() -> None:
    """Page 4 : `2 y^2 = 1 + y^2`, donc `y = +-1` m."""
    assert fl.ordonnees() == (-1, 1)


def test_q2_la_force_est_bien_selon_x() -> None:
    """En `y = 1` : aucune composante selon `y`, et `-mu0 I/(2 pi)` selon `x`."""
    selon_x, selon_y = fl.force_lineique(1)
    assert selon_y == 0
    assert sp.simplify(selon_x + fl.MU0 * fl.COURANT / (2 * sp.pi)) == 0


def test_q2_ailleurs_la_force_n_est_pas_alignee() -> None:
    _, selon_y = fl.force_lineique(2)
    assert selon_y != 0


@pytest.mark.parametrize("ordonnee", [-1.0, 1.0, 0.5])
def test_q2_par_l_integrale_de_biot_et_savart(ordonnee: float) -> None:
    """Second chemin : les deux fils integres element par element."""
    h_x, h_y = fl.champ_chiffre(ordonnee)
    par_x, par_y = fl.champ_par_biot_et_savart(ordonnee)
    assert par_x == pytest.approx(h_x, rel=1e-6, abs=1e-9)
    assert par_y == pytest.approx(h_y, rel=1e-6, abs=1e-9)


def test_q2_le_point_est_hors_des_conducteurs() -> None:
    """Rayons de 2 mm : en `y = 1` m, le champ est celui de fils sans epaisseur."""
    assert min(abs(y) for y in fl.ordonnees()) > fl.RAYON


# --- Question 3 : deux courants en phase ------------------------------------


def test_q3_le_dephasage_du_manuscrit() -> None:
    """Page 5 : `arg(I1/I2) = arctan(w L2/R2) - arctan(w L1/R1)`."""
    attendu = sp.atan(ph.W * ph.L2 / ph.R2) - sp.atan(ph.W * ph.L1 / ph.R1)
    assert sp.simplify(ph.dephasage() - attendu) == 0


def test_q3_la_condition_du_manuscrit() -> None:
    """Page 5 : `L2/R2 = L1/R1`."""
    assert ph.condition() == sp.Eq(ph.L2, ph.L1 * ph.R2 / ph.R1)


def test_q3_sous_la_condition_le_rapport_est_reel_et_positif() -> None:
    """`I1/I2 = R2/R1` : memes phases, amplitudes en raison inverse des resistances."""
    rapport = ph.rapport().subs(ph.L2, ph.L1 * ph.R2 / ph.R1)
    assert sp.simplify(rapport - ph.R2 / ph.R1) == 0


def test_q3_le_dephasage_ne_peut_pas_valoir_pi() -> None:
    """Deux arcs tangentes de nombres positifs : l'ecart reste sous `pi/2`."""
    for l1, l2 in ((1e-6, 1e6), (1e6, 1e-6)):
        valeur = ph.dephasage().subs(
            {ph.W: 1, ph.R1: 1, ph.R2: 1, ph.L1: l1, ph.L2: l2}
        )
        assert abs(float(valeur)) < math.pi / 2


@pytest.mark.parametrize(("l2", "en_phase"), [(6, True), (5, False)])
def test_q3_par_le_reseau_complet(l2: int, *, en_phase: bool) -> None:
    """Second chemin : le circuit entier, source et `R0` compris, en complexes."""
    premier, second = ph.courants_du_reseau(r0=7, r1=2, r2=4, l1=3, l2=l2, w=5)
    ecart = sp.arg(premier) - sp.arg(second)
    assert (abs(float(ecart)) < 1e-12) is en_phase


def test_q3_le_reseau_rend_bien_le_rapport_des_impedances() -> None:
    """Les courants lus dans les resistances du reseau : `I1/I2 = Z2/Z1`."""
    premier, second = ph.courants_du_reseau(r0=7, r1=2, r2=4, l1=3, l2=5, w=5)
    valeurs = {ph.R1: 2, ph.R2: 4, ph.L1: 3, ph.L2: 5, ph.W: 5}
    assert sp.simplify(premier / second - ph.rapport().subs(valeurs)) == 0


def test_q3_ni_la_source_ni_r0_n_interviennent() -> None:
    """Les deux branches voient la meme tension : `R0` ne change pas leur ecart."""
    ecarts = set()
    for r0 in (1, 50):
        premier, second = ph.courants_du_reseau(r0=r0, r1=2, r2=4, l1=3, l2=5, w=5)
        ecarts.add(sp.nsimplify(sp.simplify(premier / second)))
    assert len(ecarts) == 1
