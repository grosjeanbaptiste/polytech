"""Syllabus d'electricite generale, chapitre 15 : la branche fondamentale.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 15-2 a 15-22, lues sur les pages rendues.

Les nombres du cours : 6 000 km et 30 km de longueur d'onde (page 15-2),
et les durees reduites 2,3 ; 4,6 ; 6,9 ; 9,2 de la page 15-18. Les
fermetures de circuit sont resolues deux fois : par leur equation
differentielle, et par la transformee de Laplace que le syllabus ecrit
en (15.9.9), (15.10.8) et (15.10.9).
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.branche_fondamentale import (
    BrancheFondamentale,
    scinder_la_puissance,
    scinder_la_puissance_mutuelle,
)
from polytech.electromagnetisme.domain.circuits.fermeture import (
    charge_rc,
    courant_rl,
    regime_et_transitoire,
    transformee_de_la_charge_rc,
    transformee_du_courant_rl,
)
from polytech.electromagnetisme.domain.circuits.sinusoidaux import en_parallele
from polytech.electromagnetisme.domain.circuits.transitoires import (
    constante_de_temps_rc,
    constante_de_temps_rl,
    instant_du_seuil,
)
from polytech.electromagnetisme.domain.energie_et_forces import (
    DEPLACEMENT,
    energie_d_une_inductance,
    force_a_courant_constant,
)
from polytech.electromagnetisme.domain.onde import (
    celerite_dans_le_vide,
    longueur_d_onde,
)
from polytech.electromagnetisme.domain.quasi_stationnaire import (
    rapport_a_la_longueur_d_onde,
)
from polytech.electromagnetisme.domain.symboles import TEMPS
from polytech.electromagnetisme.domain.transitoire.laplace import (
    VARIABLE,
    original,
    valeur_finale,
    valeur_initiale,
)

E, R, L, C = sp.symbols("E R L C", positive=True)
CHARGE = sp.Function("q")(TEMPS)
COURANT = sp.diff(CHARGE, TEMPS)


def test_les_longueurs_d_onde_de_la_page_15_2() -> None:
    """« A 50 Hz, la longueur d'onde est de 6 000 km » ; a 10 kHz, 30 km."""
    celerite = celerite_dans_le_vide()
    assert longueur_d_onde(celerite, 50.0) == pytest.approx(6.0e6, rel=1e-3)
    assert longueur_d_onde(celerite, 10_000.0) == pytest.approx(30.0e3, rel=1e-3)


def test_un_systeme_petit_devant_la_longueur_d_onde_est_quasi_stationnaire() -> None:
    """Page 15-2 : `L / c_0` petit devant la periode, soit `L` devant `lambda`."""
    ligne = rapport_a_la_longueur_d_onde(100e3, 50.0)
    assert ligne == pytest.approx(100e3 / 6.0e6, rel=1e-3)
    antenne = rapport_a_la_longueur_d_onde(1.5, 100e6)
    assert antenne == pytest.approx(0.5, rel=1e-3)
    assert ligne < antenne


def test_une_dimension_ou_une_frequence_non_positive_est_refusee() -> None:
    with pytest.raises(ValueError, match="dimension nulle ou negative"):
        rapport_a_la_longueur_d_onde(0.0, 50.0)
    with pytest.raises(ValueError, match="frequence nulle ou negative"):
        rapport_a_la_longueur_d_onde(1.0, 0.0)


BRANCHE = BrancheFondamentale(fem=E, resistance=R, inductance=L, capacite=C)


def test_la_tension_de_la_branche_ajoute_ses_quatre_termes() -> None:
    """(15.5.8) `u = e + R i + L di/dt + q / C`, avec (15.5.9) `i = dq/dt`."""
    attendue = E + R * COURANT + L * sp.diff(COURANT, TEMPS) + CHARGE / C
    assert sp.simplify(BRANCHE.tension(CHARGE) - attendue) == 0


def test_sans_bobine_ni_condensateur_on_retrouve_la_loi_d_ohm() -> None:
    """(15.3.1) `u_12 = e_21 + R_12 i_12` : un condensateur absent est un fil."""
    nue = BrancheFondamentale(fem=E, resistance=R)
    assert sp.simplify(nue.tension(CHARGE) - (E + R * COURANT)) == 0


def test_une_induction_mutuelle_ajoute_m_di_prime_sur_dt() -> None:
    """(15.5.10) : la bobine voisine parcourue par `i'`."""
    mutuelle, voisin = sp.Symbol("M", real=True), sp.Function("i_p")(TEMPS)
    couplee = BRANCHE.tension(CHARGE, mutuelle=mutuelle, voisin=voisin)
    ecart = couplee - BRANCHE.tension(CHARGE)
    assert sp.simplify(ecart - mutuelle * sp.diff(voisin, TEMPS)) == 0


def test_une_capacite_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="capacite nulle"):
        BrancheFondamentale(fem=E, resistance=R, capacite=0)


def test_le_bilan_energetique_de_la_branche() -> None:
    """(15.6.1) : `u i = e i + R i^2 + d/dt (L i^2 / 2) + d/dt (C u_C^2 / 2)`."""
    source, joule, magnetique, electrique = BRANCHE.puissances(CHARGE)
    assert source == E * COURANT
    assert joule == R * COURANT**2
    assert sp.simplify(magnetique - sp.diff(L * COURANT**2 / 2, TEMPS)) == 0
    assert sp.simplify(electrique - sp.diff(CHARGE**2 / (2 * C), TEMPS)) == 0
    recue = BRANCHE.tension(CHARGE) * COURANT
    assert sp.simplify(source + joule + magnetique + electrique - recue) == 0


def test_une_bobine_deformable_travaille_en_plus_d_emmagasiner() -> None:
    """(15.6.2) : `i d(L i)/dt = d/dt (L i^2 / 2) + (1/2) i^2 dL/dt`."""
    inductance, courant = sp.Function("L")(TEMPS), sp.Function("i")(TEMPS)
    emmagasinee, travail = scinder_la_puissance(inductance, courant)
    assert emmagasinee == sp.diff(inductance * courant**2 / 2, TEMPS)
    assert travail == courant**2 * sp.diff(inductance, TEMPS) / 2
    absorbee = courant * sp.diff(inductance * courant, TEMPS)
    assert sp.simplify(emmagasinee + travail - absorbee) == 0


def test_ce_travail_est_celui_des_forces_du_champ() -> None:
    """Page 15-11 et section 11.2 : `F v`, avec `F = +(dW_m/dx)_I`."""
    courant, vitesse = sp.symbols("i v", real=True)
    inductance = sp.Function("L")(DEPLACEMENT)
    force = force_a_courant_constant(energie_d_une_inductance(inductance, courant))
    position = sp.Function("x")(TEMPS)
    mobile = inductance.subs(DEPLACEMENT, position)
    _, travail = scinder_la_puissance(mobile, courant)
    mecanique = (force * vitesse).subs(DEPLACEMENT, position)
    assert sp.simplify(travail.subs(sp.diff(position, TEMPS), vitesse) - mecanique) == 0


def test_un_condensateur_deformable_suit_la_meme_regle() -> None:
    """(15.6.3) : `u_C d(C u_C)/dt = d/dt (C u_C^2 / 2) + (1/2) u_C^2 dC/dt`."""
    capacite, tension = sp.Function("C")(TEMPS), sp.Function("u_C")(TEMPS)
    emmagasinee, travail = scinder_la_puissance(capacite, tension)
    absorbee = tension * sp.diff(capacite * tension, TEMPS)
    assert sp.simplify(emmagasinee + travail - absorbee) == 0


def test_deux_bobines_mobiles_se_partagent_l_energie_mutuelle() -> None:
    """(15.6.4) : `i d(M i')/dt + i' d(M i)/dt = d/dt (M i i') + i i' dM/dt`."""
    mutuelle = sp.Function("M")(TEMPS)
    premier, second = sp.Function("i")(TEMPS), sp.Function("i_p")(TEMPS)
    emmagasinee, travail = scinder_la_puissance_mutuelle(mutuelle, premier, second)
    assert emmagasinee == sp.diff(mutuelle * premier * second, TEMPS)
    assert travail == premier * second * sp.diff(mutuelle, TEMPS)
    absorbee = premier * sp.diff(mutuelle * second, TEMPS)
    absorbee += second * sp.diff(mutuelle * premier, TEMPS)
    assert sp.simplify(emmagasinee + travail - absorbee) == 0


def test_la_fermeture_du_circuit_rl() -> None:
    """(15.9.1) `E = R i + L di/dt`, (15.9.3) `i = (E / R)(1 - exp(-t / tau))`."""
    courant = courant_rl(E, R, L)
    assert sp.simplify(R * courant + L * sp.diff(courant, TEMPS) - E) == 0
    # (15.8.2) : le courant d'une bobine ne saute pas.
    assert courant.subs(TEMPS, 0) == 0
    assert sp.limit(courant, TEMPS, sp.oo) == E / R
    # (15.9.4) et (15.9.5) : les deux tensions reduites se completent.
    sur_la_resistance = R * courant / E
    sur_la_bobine = L * sp.diff(courant, TEMPS) / E
    assert sp.simplify(sur_la_bobine - sp.exp(-TEMPS * R / L)) == 0
    assert sp.simplify(sur_la_resistance + sur_la_bobine - 1) == 0


def test_la_pente_initiale_atteindrait_le_regime_en_une_constante_de_temps() -> None:
    """(15.9.7) : `(di/dt)_0 = I / tau`, la tangente de la figure 15.9.2."""
    pente = sp.diff(courant_rl(E, R, L), TEMPS).subs(TEMPS, 0)
    assert sp.simplify(pente - (E / R) / (L / R)) == 0
    assert constante_de_temps_rl(4.0, 2.0) == pytest.approx(0.5)


def test_les_durees_reduites_de_la_page_15_18() -> None:
    """(15.9.6) : `t / tau = ln(1 / (1 - i / I))`, soit 2,3 ; 4,6 ; 6,9 ; 9,2."""
    lus = (2.3, 4.6, 6.9, 9.2)
    for fraction, attendu in zip((0.9, 0.99, 0.999, 0.9999), lus, strict=True):
        duree = float(instant_du_seuil(1.0, 1.0, fraction))
        assert duree == pytest.approx(attendu, abs=0.02)
    # La regle de l'art : le regime est atteint apres cinq constantes de temps.
    assert 1 - math.exp(-5) > 0.99


def test_laplace_retrouve_le_courant_du_circuit_rl() -> None:
    """(15.9.9) : `I(p) = E / (p (p L + R)) = (E / R)(1 / p - 1 / (p + 1 / tau))`."""
    transformee = transformee_du_courant_rl(E, R, L)
    simples = (E / R) * (1 / VARIABLE - 1 / (VARIABLE + R / L))
    assert sp.simplify(transformee - simples) == 0
    assert sp.simplify(original(transformee) - courant_rl(E, R, L)) == 0
    assert valeur_initiale(transformee) == 0
    assert valeur_finale(transformee) == E / R


def test_la_fermeture_du_circuit_rc() -> None:
    """(15.10.2) `E = R dq/dt + q / C`, (15.10.3) `q = C E (1 - exp(-t / tau))`."""
    charge = charge_rc(E, R, C)
    assert sp.simplify(R * sp.diff(charge, TEMPS) + charge / C - E) == 0
    # (15.8.1) : la charge d'un condensateur ne saute pas.
    assert charge.subs(TEMPS, 0) == 0
    # Page 15-20 : le courant, lui, saute a `E / R`.
    courant = sp.diff(charge, TEMPS)
    assert sp.simplify(courant.subs(TEMPS, 0) - E / R) == 0
    # (15.10.5) et (15.10.6).
    assert sp.simplify(R * courant / E - sp.exp(-TEMPS / (R * C))) == 0
    assert sp.simplify(charge / (C * E) + R * courant / E - 1) == 0
    assert constante_de_temps_rc(1e3, 1e-6) == pytest.approx(1e-3)


def test_un_condensateur_deja_charge_part_de_sa_charge() -> None:
    """(15.10.10) : `q = C E - (C E - Q_0) exp(-t / tau)`."""
    initiale = sp.Symbol("Q_0", real=True)
    charge = charge_rc(E, R, C, initiale=initiale)
    assert charge.subs(TEMPS, 0) == initiale
    assert sp.simplify(R * sp.diff(charge, TEMPS) + charge / C - E) == 0
    assert sp.limit(charge, TEMPS, sp.oo) == C * E


def test_laplace_retrouve_la_charge_et_le_courant_du_circuit_rc() -> None:
    """(15.10.8) `Q(p) = C E (1 / p - 1 / (p + 1 / tau))`, (15.10.9) `I(p) = p Q(p)`."""
    transformee = transformee_de_la_charge_rc(E, R, C)
    simples = C * E * (1 / VARIABLE - 1 / (VARIABLE + 1 / (R * C)))
    assert sp.simplify(transformee - simples) == 0
    assert sp.simplify(original(transformee) - charge_rc(E, R, C)) == 0
    du_courant = VARIABLE * transformee
    assert sp.simplify(du_courant - (E / R) / (VARIABLE + 1 / (R * C))) == 0
    assert valeur_initiale(du_courant) == E / R


def test_le_regime_et_le_phenomene_transitoire_se_separent() -> None:
    """15.11 : la solution particuliere reste, la solution homogene s'eteint."""
    regime, transitoire = regime_et_transitoire(courant_rl(E, R, L))
    assert regime == E / R
    assert sp.limit(transitoire, TEMPS, sp.oo) == 0
    assert sp.simplify(R * transitoire + L * sp.diff(transitoire, TEMPS)) == 0


def test_des_inductances_en_parallele_s_associent_comme_des_resistances() -> None:
    """Note de la page 15-16 : `1 / L = 1 / L_1 + 1 / L_2`, sans mutuelle."""
    assert en_parallele(sp.Integer(6), sp.Integer(3)) == 2
