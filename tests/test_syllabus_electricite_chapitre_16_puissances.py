"""Syllabus d'electricite generale, chapitre 16 : puissances et enclenchement.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 16-22 a 16-37, lues sur les pages rendues.

Les nombres du cours : 70 et 60 degres pour la figure 16.13.1 ; seize et
cent soixante periodes, soit 0,32 s et 3,2 s a 50 Hz, pour l'erreur d'un
pour cent de la page 16-26. La conservation des puissances active et
reactive est eprouvee sur un reseau resolu par la methode des mailles.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.branche_sinusoidale import (
    Nature,
    admittance,
    impedance_de_la_branche,
)
from polytech.electromagnetisme.domain.circuits.enclenchement_sinusoidal import (
    courant_a_l_enclenchement_rl,
)
from polytech.electromagnetisme.domain.circuits.grandeur_sinusoidale import (
    GrandeurSinusoidale,
)
from polytech.electromagnetisme.domain.circuits.mailles import (
    Branche,
    resoudre_par_les_mailles,
)
from polytech.electromagnetisme.domain.circuits.periodiques import (
    courant_actif,
    courant_reactif,
    puissance_apparente,
    puissance_reactive_globale,
)
from polytech.electromagnetisme.domain.circuits.phaseurs import puissance_complexe
from polytech.electromagnetisme.domain.circuits.puissances_sinusoidales import (
    Role,
    composantes_de_la_puissance,
    courants_actif_et_reactif,
    erreur_relative_maximale,
    pertes_en_ligne,
    puissance_active,
    puissance_instantanee,
    puissance_reactive,
    puissance_reactive_rlc,
    role_et_nature,
)
from polytech.electromagnetisme.domain.symboles import TEMPS

OMEGA = sp.Symbol("omega", positive=True)
U, EFF = sp.symbols("U I", positive=True)
THETA, PHI = sp.symbols("theta_u varphi", real=True)
TENSION = GrandeurSinusoidale(efficace=U, phase=sp.pi / 4, pulsation=OMEGA)
COURANT = GrandeurSinusoidale(
    efficace=EFF, phase=sp.pi / 4 - sp.pi / 3, pulsation=OMEGA
)
PERIODE = 2 * sp.pi / OMEGA


def test_la_puissance_instantanee_est_le_produit_u_i() -> None:
    """(16.13.1) `p = u i`, developpee en (16.13.2)."""
    produit = TENSION.instantanee() * COURANT.instantanee()
    formule = puissance_instantanee(U, EFF, sp.pi / 4, sp.pi / 3, OMEGA)
    assert sp.simplify(produit - formule) == 0


def test_la_puissance_moyenne_est_u_i_cos_phi() -> None:
    """(16.13.3) integree sur une periode donne (16.13.4)."""
    instantanee = puissance_instantanee(U, EFF, THETA, PHI, OMEGA)
    moyenne = sp.integrate(instantanee, (TEMPS, 0, PERIODE)) / PERIODE
    assert sp.simplify(moyenne - puissance_active(U, EFF, PHI)) == 0
    assert puissance_active(U, EFF, PHI) == U * EFF * sp.cos(PHI)


def test_la_puissance_fluctuante_de_la_figure_16_13_1() -> None:
    """`theta_u = 70`, `phi = 60` degres : `P = U I / 2`, creux a `-U I / 2`.

    Le terme fluctuant est de pulsation `2 omega` : la figure cote `T / 2`.
    """
    degre = sp.pi / 180
    instantanee = puissance_instantanee(1, 1, 70 * degre, 60 * degre, 1)
    assert puissance_active(1, 1, 60 * degre) == sp.Rational(1, 2)
    fluctuante = instantanee - sp.Rational(1, 2)
    assert sp.simplify(fluctuante.subs(TEMPS, TEMPS + sp.pi) - fluctuante) == 0
    assert sp.minimum(instantanee, TEMPS, sp.Interval(0, 2 * sp.pi)) == (
        -sp.Rational(1, 2)
    )


def test_une_resistance_dissipe_r_i_carre_et_une_reactance_rien() -> None:
    """(16.13.6) `P = R I^2` ; (16.13.7) `P = 0` pour une bobine ou une capacite."""
    resistance = sp.Symbol("R", positive=True)
    assert puissance_active(resistance * EFF, EFF, 0) == resistance * EFF**2
    assert puissance_active(U, EFF, sp.pi / 2) == 0
    assert puissance_active(U, EFF, -sp.pi / 2) == 0


def test_l_energie_d_une_bobine_oscille_a_deux_omega() -> None:
    """(16.13.8) : `L i^2 / 2 = (L I^2 / 2) [1 + cos 2(omega t + theta_u - phi)]`."""
    inductance = sp.Symbol("L", positive=True)
    courant = GrandeurSinusoidale(efficace=EFF, phase=sp.pi / 6, pulsation=OMEGA)
    energie = inductance * courant.instantanee() ** 2 / 2
    attendue = inductance * EFF**2 / 2 * (1 + sp.cos(2 * (OMEGA * TEMPS + sp.pi / 6)))
    assert sp.simplify(energie - attendue) == 0


def test_l_energie_absorbee_et_son_terme_correctif() -> None:
    """(16.14.1), ou `t_2 = t_1 + n T + t'`, contre l'integrale de `p`."""
    debut, fraction = sp.Rational(3, 10), sp.Rational(7, 10)
    theta, phi = sp.pi / 5, sp.pi / 7
    fin = debut + 3 * 2 * sp.pi + fraction
    instantanee = puissance_instantanee(1, 1, theta, phi, 1)
    integrale = sp.integrate(instantanee, (TEMPS, debut, fin))
    correction = sp.cos(2 * debut + fraction + 2 * theta - phi) * sp.sin(fraction)
    correction *= 2 * sp.pi / (fin - debut) / (2 * sp.pi) / sp.cos(phi)
    formule = sp.cos(phi) * (fin - debut) * (1 + correction)
    assert float(integrale) == pytest.approx(float(formule))


def test_l_erreur_tombe_sous_un_pour_cent_apres_seize_periodes() -> None:
    """(16.14.3) et page 16-26 : 16 periodes si `cos(phi) = 1`, 160 si 0,1."""
    assert erreur_relative_maximale(16, 1.0) < 0.01 < erreur_relative_maximale(15, 1.0)
    assert erreur_relative_maximale(160, 0.1) < 0.01
    assert erreur_relative_maximale(160, -0.1) == erreur_relative_maximale(160, 0.1)
    # A 50 Hz, la periode vaut 20 ms : 0,32 s et 3,2 s.
    periode = 0.02
    assert periode * 16 == pytest.approx(0.32)
    assert periode * 160 == pytest.approx(3.2)


def test_un_intervalle_ou_un_facteur_de_puissance_nul_n_a_pas_de_borne() -> None:
    with pytest.raises(ValueError, match="duree nulle ou negative"):
        erreur_relative_maximale(0, 1.0)
    with pytest.raises(ValueError, match="facteur de puissance nul"):
        erreur_relative_maximale(16, 0.0)


def test_la_puissance_reactive_d_une_bobine_et_d_une_capacite() -> None:
    """(16.15.2) `Q = U I sin(phi)` ; (16.15.4) `omega L I^2` ; (16.15.5) negative."""
    inductance, capacite = sp.symbols("L C", positive=True)
    bobine = puissance_reactive(OMEGA * inductance * EFF, EFF, sp.pi / 2)
    assert bobine == OMEGA * inductance * EFF**2
    condensateur = puissance_reactive(EFF / (OMEGA * capacite), EFF, -sp.pi / 2)
    assert condensateur == -(EFF**2) / (OMEGA * capacite)
    assert puissance_reactive(U, EFF, 0) == 0


def test_la_puissance_reactive_compare_les_deux_energies_emmagasinees() -> None:
    """(16.15.3) : `Q = 2 omega (L I^2 / 2 - Q_c^2 / (2 C))`, avec `I = omega Q_c`."""
    inductance, capacite = sp.symbols("L C", positive=True)
    reactive = puissance_reactive_rlc(OMEGA, inductance, capacite, EFF)
    par_la_reactance = (OMEGA * inductance - 1 / (OMEGA * capacite)) * EFF**2
    assert sp.simplify(reactive - par_la_reactance) == 0
    charge = EFF / OMEGA
    energies = inductance * EFF**2 / 2 - charge**2 / (2 * capacite)
    assert sp.simplify(reactive - 2 * OMEGA * energies) == 0


def test_la_puissance_complexe_d_une_branche_rlc() -> None:
    """(16.17.3) `S = U I* = P + j Q`, (16.17.7) `R I^2 + j X I^2`."""
    impedance = impedance_de_la_branche(3, 2, sp.Rational(1, 2), 2)
    courant = 2 * sp.exp(sp.I * sp.pi / 5)
    complexe = sp.simplify(puissance_complexe(impedance * courant, courant))
    assert complexe == 12 + 12 * sp.I
    assert sp.im(complexe) == puissance_reactive_rlc(2, 2, sp.Rational(1, 2), 2)
    # (16.16.2) et (16.17.4) : `S = racine(P^2 + Q^2) = |S|`.
    apparente = puissance_apparente(float(sp.Abs(impedance * courant)), 2.0)
    assert apparente == pytest.approx(float(sp.Abs(complexe)))
    assert puissance_reactive_globale(apparente, 12.0) == pytest.approx(12.0)


def test_la_puissance_se_decompose_en_active_et_reactive_instantanees() -> None:
    """(16.15.8) : `p = P + P cos 2(omega t + theta_u) + Q sin 2(omega t + theta_u)`."""
    active = puissance_active(U, EFF, PHI)
    reactive = puissance_reactive(U, EFF, PHI)
    termes = composantes_de_la_puissance(active, reactive, THETA, OMEGA)
    assert termes[0] == active
    instantanee = puissance_instantanee(U, EFF, THETA, PHI, OMEGA)
    assert sp.simplify(sp.expand_trig(sum(termes) - instantanee)) == 0


def test_la_puissance_instantanee_par_la_puissance_complexe() -> None:
    """(16.17.6) : `p = Re{S [1 + exp(j 2 (omega t + theta_u - phi))]}`."""
    theta, phi = sp.pi / 5, sp.pi / 7
    complexe = 6 * sp.exp(sp.I * phi)
    tournant = complexe * (1 + sp.exp(2 * sp.I * (OMEGA * TEMPS + theta - phi)))
    partie_reelle = sp.re(sp.expand_complex(tournant))
    attendue = puissance_instantanee(2, 3, theta, phi, OMEGA)
    assert sp.simplify(partie_reelle - attendue) == 0


QUADRIPOLE = (
    Branche("0", "1", 0, sp.Integer(10)),
    Branche("1", "m", 3 + 4 * sp.I),
    Branche("m", "0", -5 * sp.I),
    Branche("m", "2", sp.Integer(2)),
    Branche("2", "0", 1 + sp.I),
)
"""Une source, trois impedances en T et une charge."""


def _puissances_des_branches() -> list[sp.Expr]:
    """`U I*` de chaque branche, la tension etant `Z I - E` de `de` vers `vers`."""
    courants = resoudre_par_les_mailles(QUADRIPOLE, (1, 2, 3))
    return [
        sp.simplify(puissance_complexe(b.resistance * i - b.fem, i))
        for b, i in zip(QUADRIPOLE, courants, strict=True)
    ]


def test_les_puissances_active_et_reactive_se_conservent() -> None:
    """(16.15.1) et page 16-31 : la somme des puissances complexes est nulle."""
    puissances = _puissances_des_branches()
    assert sp.simplify(sum(puissances)) == 0
    # La source fournit : sa puissance active recue est negative.
    assert sp.re(puissances[0]) < 0


def test_la_puissance_apparente_ne_se_conserve_pas() -> None:
    """Page 16-30 : elle « n'obeit a aucun principe de conservation »."""
    puissances = _puissances_des_branches()
    fournie = sp.Abs(puissances[0])
    recues = sum(sp.Abs(s) for s in puissances[1:])
    assert float(recues) > float(fournie) * 1.01


def test_les_quatre_quadrants_de_la_figure_16_18_1() -> None:
    """Recepteur ou generateur selon `P`, inductif ou capacitif selon `Q`."""
    assert role_et_nature(5.0, 2.0) == (Role.RECEPTEUR, Nature.INDUCTIVE)
    assert role_et_nature(5.0, -2.0) == (Role.RECEPTEUR, Nature.CAPACITIVE)
    assert role_et_nature(-5.0, 2.0) == (Role.GENERATEUR, Nature.INDUCTIVE)
    assert role_et_nature(-5.0, -2.0) == (Role.GENERATEUR, Nature.CAPACITIVE)
    assert role_et_nature(5.0, 0.0) == (Role.RECEPTEUR, Nature.RESONANTE)


def test_une_branche_sans_puissance_active_n_a_pas_de_role() -> None:
    with pytest.raises(ValueError, match="puissance active nulle"):
        role_et_nature(0.0, 2.0)


def test_les_courants_actif_et_reactif() -> None:
    """(16.19.4) a (16.19.8) : `I_w = I cos(phi)`, `I_d = I sin(phi)`."""
    actif, reactif = courants_actif_et_reactif(10.0, math.pi / 3)
    assert actif == pytest.approx(5.0)
    assert reactif == pytest.approx(5.0 * math.sqrt(3))
    assert actif**2 + reactif**2 == pytest.approx(100.0)
    active = float(puissance_active(230, 10, sp.pi / 3))
    reactive = float(puissance_reactive(230, 10, sp.pi / 3))
    assert courant_actif(active, 230.0) == pytest.approx(actif)
    assert courant_reactif(reactive, 230.0) == pytest.approx(reactif)


def test_les_courants_actif_et_reactif_par_l_admittance() -> None:
    """(16.19.13) : `I_w = G U` ; `I_d` de substitut `j B U`, soit `-B U` en retard."""
    impedance = impedance_de_la_branche(3, 2, sp.oo, 2)
    conductance, susceptance = admittance(impedance)
    courant, dephasage = 2.0, math.atan2(4, 3)
    actif, reactif = courants_actif_et_reactif(courant, dephasage)
    assert float(conductance * 10) == pytest.approx(actif)
    assert float(-susceptance * 10) == pytest.approx(reactif)


def test_les_pertes_en_ligne_croissent_quand_le_facteur_de_puissance_baisse() -> None:
    """Exemple 16.19.1, (16.19.14) : `P_l = R (I_w^2 + I_d^2)`, `I_w` impose."""
    ideal = pertes_en_ligne(0.5, puissance=2300.0, tension=230.0, facteur=1.0)
    assert ideal == pytest.approx(0.5 * 10.0**2)
    moitie = pertes_en_ligne(0.5, puissance=2300.0, tension=230.0, facteur=0.5)
    assert moitie == pytest.approx(4 * ideal)
    actif, reactif = courants_actif_et_reactif(20.0, math.pi / 3)
    assert moitie == pytest.approx(0.5 * (actif**2 + reactif**2))


def test_un_facteur_de_puissance_hors_de_zero_un_est_refuse() -> None:
    with pytest.raises(ValueError, match="facteur de puissance hors de"):
        pertes_en_ligne(0.5, puissance=2300.0, tension=230.0, facteur=0.0)
    with pytest.raises(ValueError, match="facteur de puissance hors de"):
        pertes_en_ligne(0.5, puissance=2300.0, tension=230.0, facteur=1.2)


R, L = sp.symbols("R L", positive=True)


def test_l_enclenchement_d_un_circuit_rl_sur_une_tension_sinusoidale() -> None:
    """(16.20.1) `u = R i + L di/dt`, (16.20.8), et `i(0) = 0` par (15.8.2)."""
    courant = courant_a_l_enclenchement_rl(U, THETA, R, L, OMEGA)
    tension = U * sp.sqrt(2) * sp.cos(OMEGA * TEMPS + THETA)
    residu = R * courant + L * sp.diff(courant, TEMPS) - tension
    assert sp.simplify(sp.expand_trig(residu)) == 0
    assert sp.simplify(courant.subs(TEMPS, 0)) == 0


def test_enclenche_au_bon_instant_le_circuit_n_a_pas_de_transitoire() -> None:
    """Page 16-37 : `cos(theta_u - phi) = 0`, ici `R = omega L` et `phi = pi / 4`."""
    courant = courant_a_l_enclenchement_rl(10, 3 * sp.pi / 4, 2, 1, 2)
    efficace = 10 / sp.sqrt(8)
    regime = efficace * sp.sqrt(2) * sp.cos(2 * TEMPS + sp.pi / 2)
    assert sp.simplify(courant - regime) == 0


def test_enclenche_au_pire_instant_le_courant_approche_le_double() -> None:
    """Page 16-37 : `|i| = I racine(2) (1 + exp(-T / (2 tau)))` a l'instant `T / 2`."""
    courant = courant_a_l_enclenchement_rl(10, sp.pi / 4, 2, 1, 2)
    efficace, periode, tau = 10 / sp.sqrt(8), sp.pi, sp.Rational(1, 2)
    a_la_demi_periode = sp.Abs(courant.subs(TEMPS, periode / 2))
    attendu = efficace * sp.sqrt(2) * (1 + sp.exp(-periode / (2 * tau)))
    assert sp.simplify(a_la_demi_periode - attendu) == 0
