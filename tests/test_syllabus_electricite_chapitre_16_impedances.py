"""Syllabus d'electricite generale, chapitre 16 : substituts complexes et impedances.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 16-2 a 16-22, lues sur les pages rendues.

La methode des nombres complexes est eprouvee contre ce qu'elle remplace :
l'equation differentielle de la branche fondamentale du chapitre 15, et
la methode des mailles du chapitre 13 ou les impedances prennent la place
des resistances. La figure 16.5.1 donne le seul oracle chiffre : -60 et
70 degres, soit 130 degres de dephasage.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.branche_fondamentale import (
    BrancheFondamentale,
)
from polytech.electromagnetisme.domain.circuits.branche_sinusoidale import (
    Nature,
    admittance,
    capacite_apparente,
    courant_de_la_branche,
    impedance_de_la_branche,
    inductance_apparente,
    module_et_argument,
    nature_de_la_branche,
    reactance,
)
from polytech.electromagnetisme.domain.circuits.grandeur_sinusoidale import (
    GrandeurSinusoidale,
    RelationDePhase,
    avance,
    dephasage,
    relation_de_phase,
    substitut_de_la_derivee,
)
from polytech.electromagnetisme.domain.circuits.mailles import (
    Branche,
    resoudre_par_les_mailles,
)
from polytech.electromagnetisme.domain.circuits.phaseurs import valeur_instantanee
from polytech.electromagnetisme.domain.circuits.sinusoidaux import (
    en_parallele,
    en_serie,
    impedance_capacitive,
    impedance_inductive,
    impedance_resistive,
)
from polytech.electromagnetisme.domain.circuits.theoremes import (
    conductance_de_transfert,
)
from polytech.electromagnetisme.domain.circuits.valeurs_efficaces import (
    valeur_efficace,
)
from polytech.electromagnetisme.domain.symboles import PULSATION, TEMPS

OMEGA = sp.Symbol("omega", positive=True)
GRANDEUR = GrandeurSinusoidale(efficace=sp.Integer(5), phase=sp.pi / 3, pulsation=OMEGA)


def test_periode_frequence_et_pulsation_du_reseau_a_50_hertz() -> None:
    """(16.2.2) a (16.2.4) : `T = 2 pi / omega`, `f = 1 / T`, `omega = 2 pi f`."""
    reseau = GrandeurSinusoidale(efficace=230, phase=0, pulsation=100 * sp.pi)
    assert reseau.periode() == sp.Rational(1, 50)
    assert reseau.frequence() == 50
    assert reseau.amplitude() == 230 * sp.sqrt(2)


def test_la_valeur_efficace_est_l_amplitude_sur_racine_de_deux() -> None:
    """(16.3.4) integree donne (16.3.6) : `X = x_m / racine(2)`."""
    quadratique = valeur_efficace(GRANDEUR.instantanee(), GRANDEUR.periode())
    assert sp.simplify(quadratique - 5) == 0


def test_le_substitut_complexe_redonne_la_valeur_instantanee() -> None:
    """(16.4.3) `X exp(j theta)`, et (16.4.6) qui en redonne `x(t)`."""
    assert GRANDEUR.substitut() == 5 * sp.exp(sp.I * sp.pi / 3)
    retour = valeur_instantanee(GRANDEUR.substitut(), OMEGA)
    assert sp.simplify(retour - GRANDEUR.instantanee()) == 0
    attendu = 5 * sp.sqrt(2) * sp.cos(OMEGA * TEMPS + sp.pi / 3)
    assert sp.simplify(GRANDEUR.instantanee() - attendu) == 0


def test_multiplier_par_j_fait_tourner_d_un_quart_de_tour_en_avant() -> None:
    """Page 16-7 : `j X = X exp(j (theta + pi / 2))`."""
    tourne = valeur_instantanee(sp.I * GRANDEUR.substitut(), OMEGA)
    attendu = 5 * sp.sqrt(2) * sp.cos(OMEGA * TEMPS + sp.pi / 3 + sp.pi / 2)
    assert sp.simplify(tourne - attendu) == 0


def test_deriver_revient_a_multiplier_par_j_omega() -> None:
    """(16.4.7), aux ordres un et deux."""
    for ordre in (1, 2):
        substitut = substitut_de_la_derivee(GRANDEUR.substitut(), OMEGA, ordre)
        derivee = sp.diff(GRANDEUR.instantanee(), TEMPS, ordre)
        assert sp.simplify(valeur_instantanee(substitut, OMEGA) - derivee) == 0


def test_un_quart_de_periode_echange_partie_reelle_et_imaginaire() -> None:
    """(16.4.8) : `Re{X exp(j omega (t - T / 4))} = Im{X exp(j omega t)}`."""
    substitut, quart = GRANDEUR.substitut(), GRANDEUR.periode() / 4
    retarde = sp.re(
        sp.expand_complex(substitut * sp.exp(sp.I * OMEGA * (TEMPS - quart)))
    )
    imaginaire = sp.im(sp.expand_complex(substitut * sp.exp(sp.I * OMEGA * TEMPS)))
    assert sp.simplify(retarde - imaginaire) == 0


def test_les_gardes_d_une_grandeur_sinusoidale() -> None:
    with pytest.raises(ValueError, match="valeur efficace negative"):
        GrandeurSinusoidale(efficace=-1, phase=0, pulsation=OMEGA)
    with pytest.raises(ValueError, match="phase initiale hors de"):
        GrandeurSinusoidale(efficace=1, phase=4, pulsation=OMEGA)
    with pytest.raises(ValueError, match="pulsation nulle ou negative"):
        GrandeurSinusoidale(efficace=1, phase=0, pulsation=0)


def test_le_dephasage_de_la_figure_16_5_1() -> None:
    """`theta_x = -60`, `theta_y = 70` : `y` est en avance de 130 degres sur `x`."""
    degre = sp.pi / 180
    assert dephasage(70 * degre, sur=-60 * degre) == 130 * degre
    assert dephasage(-60 * degre, sur=70 * degre) == -130 * degre


def test_le_dephasage_est_ramene_entre_moins_pi_et_pi() -> None:
    """(16.5.1) : `theta_y - theta_x + 2 k pi`, avec `k = 1, 0 ou -1`."""
    degre = sp.pi / 180
    assert dephasage(170 * degre, sur=-170 * degre) == -20 * degre
    assert dephasage(-170 * degre, sur=170 * degre) == 20 * degre
    assert dephasage(sp.pi / 2, sur=-sp.pi / 2) == sp.pi


def test_phase_quadrature_et_opposition() -> None:
    """Page 16-9 : zero, `pi / 2` (soit `T / 4`), `pi` (soit `T / 2`)."""
    assert relation_de_phase(0) is RelationDePhase.EN_PHASE
    assert relation_de_phase(-sp.pi / 2) is RelationDePhase.EN_QUADRATURE
    assert relation_de_phase(sp.pi) is RelationDePhase.EN_OPPOSITION
    assert relation_de_phase(sp.pi / 3) is RelationDePhase.QUELCONQUE
    periode = GRANDEUR.periode()
    assert sp.simplify(avance(sp.pi / 2, OMEGA) - periode / 4) == 0
    assert sp.simplify(avance(sp.pi, OMEGA) - periode / 2) == 0


R, L, C = sp.symbols("R L C", positive=True)


def test_l_impedance_complexe_de_la_branche_rlc() -> None:
    """(16.7.5) : `Z = R + j omega L + 1 / (j omega C) = R + j X`."""
    impedance = impedance_de_la_branche(R, L, C, OMEGA)
    assert sp.simplify(impedance - (R + sp.I * reactance(L, C, OMEGA))) == 0
    assert reactance(L, C, OMEGA) == OMEGA * L - 1 / (OMEGA * C)
    par_association = en_serie(
        impedance_resistive(R), impedance_inductive(L), impedance_capacitive(C)
    )
    assert sp.simplify(impedance.subs(OMEGA, PULSATION) - par_association) == 0


def test_la_tension_complexe_resout_l_equation_de_la_branche() -> None:
    """(16.7.4) `U = Z I` contre (16.7.1) `u = L di/dt + R i + q / C`.

    La charge a pour substitut `Q = I / (j omega)` (16.7.3b).
    """
    resistance, inductance, capacite, pulsation = 3, 2, sp.Rational(1, 4), 2
    courant = 5 * sp.exp(sp.I * sp.pi / 6)
    impedance = impedance_de_la_branche(resistance, inductance, capacite, pulsation)
    par_les_complexes = valeur_instantanee(impedance * courant, pulsation)
    charge = valeur_instantanee(courant / (sp.I * pulsation), pulsation)
    branche = BrancheFondamentale(0, resistance, inductance, capacite)
    assert sp.simplify(branche.tension(charge) - par_les_complexes) == 0


def test_le_module_et_l_argument_de_l_impedance() -> None:
    """(16.7.6) et (16.7.8), puis `R = Z cos(psi)` et `X = Z sin(psi)`."""
    impedance = impedance_de_la_branche(3, 2, sp.oo, 2)
    module, argument = module_et_argument(impedance)
    assert module == 5
    assert sp.simplify(sp.tan(argument) - sp.Rational(4, 3)) == 0
    assert sp.simplify(module * sp.cos(argument) - 3) == 0
    assert sp.simplify(module * sp.sin(argument) - 4) == 0


def test_une_impedance_nulle_n_a_pas_d_argument() -> None:
    with pytest.raises(ValueError, match="impedance nulle"):
        module_et_argument(sp.Integer(0))


def test_les_sept_cas_de_la_discussion_16_11() -> None:
    """De la resistance seule a la branche RLC : l'argument `psi` de chacune."""
    seule = module_et_argument(impedance_de_la_branche(3, 0, sp.oo, 2))[1]
    bobine = module_et_argument(impedance_de_la_branche(0, 2, sp.oo, 2))[1]
    capacite = module_et_argument(impedance_de_la_branche(0, 0, 1, 2))[1]
    avec_bobine = module_et_argument(impedance_de_la_branche(3, 2, sp.oo, 2))[1]
    avec_capacite = module_et_argument(impedance_de_la_branche(3, 0, 1, 2))[1]
    assert (seule, bobine, capacite) == (0, sp.pi / 2, -sp.pi / 2)
    assert 0 < avec_bobine < sp.pi / 2
    assert -sp.pi / 2 < avec_capacite < 0
    inductive = module_et_argument(impedance_de_la_branche(0, 2, 1, 2))[1]
    capacitive = module_et_argument(
        impedance_de_la_branche(0, 2, 1, sp.Rational(1, 2))
    )[1]
    assert (inductive, capacitive) == (sp.pi / 2, -sp.pi / 2)
    # 16.11.6 : a la resonance, bobine et condensateur seuls font `Z = 0`.
    assert impedance_de_la_branche(0, 2, sp.Rational(1, 8), 2) == 0
    generale = module_et_argument(impedance_de_la_branche(3, 2, 1, 2))[1]
    assert -sp.pi / 2 <= generale <= sp.pi / 2


def test_sans_force_electromotrice_le_dephasage_est_l_argument() -> None:
    """(16.7.12) : `phi = theta_u - theta_i = psi`."""
    impedance = impedance_de_la_branche(3, 2, sp.oo, 2)
    tension = 10 * sp.exp(sp.I * sp.pi / 4)
    courant = courant_de_la_branche(tension, impedance)
    assert sp.Abs(courant) == 2
    ecart = sp.arg(tension) - sp.arg(courant)
    argument = module_et_argument(impedance)[1]
    assert float(ecart - argument) == pytest.approx(0.0, abs=1e-12)


def test_avec_une_force_electromotrice_ce_lien_disparait() -> None:
    """(16.7.13) `U = E + Z I` ; 16.11.8 : « pas de lien direct entre phi et psi »."""
    impedance = impedance_de_la_branche(3, 2, sp.oo, 2)
    tension, fem = sp.Integer(10), 10 * sp.I
    courant = courant_de_la_branche(tension, impedance, fem=fem)
    assert sp.simplify(fem + impedance * courant - tension) == 0
    ecart = sp.arg(tension) - sp.arg(courant)
    argument = module_et_argument(impedance)[1]
    assert abs(float(ecart - argument)) > 0.5


def test_inductance_et_capacite_apparentes_gardent_la_reactance() -> None:
    """(16.9.1) si `omega L > 1 / (omega C)`, (16.9.2) sinon ; la resonance entre."""
    apparente = inductance_apparente(2, 1, 2)
    assert apparente == sp.Rational(7, 4)
    assert reactance(apparente, sp.oo, 2) == reactance(2, 1, 2)
    assert nature_de_la_branche(reactance(2, 1, 2)) is Nature.INDUCTIVE
    pulsation = sp.Rational(1, 2)
    equivalente = capacite_apparente(2, 1, pulsation)
    assert equivalente == 2
    assert reactance(0, equivalente, pulsation) == reactance(2, 1, pulsation)
    assert nature_de_la_branche(reactance(2, 1, pulsation)) is Nature.CAPACITIVE
    assert nature_de_la_branche(reactance(2, sp.Rational(1, 8), 2)) is Nature.RESONANTE


def test_une_branche_capacitive_n_a_pas_d_inductance_apparente() -> None:
    with pytest.raises(ValueError, match="branche non inductive"):
        inductance_apparente(2, 1, sp.Rational(1, 2))
    with pytest.raises(ValueError, match="branche non capacitive"):
        capacite_apparente(2, 1, 2)


def test_les_inductances_s_associent_comme_les_impedances() -> None:
    """(16.10.1) et (16.10.2) avec `R = 0` et `C` infinie."""
    bobines = [impedance_inductive(sp.Integer(valeur)) for valeur in (6, 3)]
    assert sp.simplify(en_serie(*bobines) - impedance_inductive(sp.Integer(9))) == 0
    assert sp.simplify(en_parallele(*bobines) - impedance_inductive(sp.Integer(2))) == 0


def test_la_conductance_et_la_susceptance() -> None:
    """(16.12.1) a (16.12.5) : `G = R / Z^2`, `B = -X / Z^2`, `psi' = -psi`."""
    impedance = impedance_de_la_branche(3, 2, sp.oo, 2)
    conductance, susceptance = admittance(impedance)
    assert (conductance, susceptance) == (sp.Rational(3, 25), -sp.Rational(4, 25))
    complexe = conductance + sp.I * susceptance
    assert sp.simplify(complexe * impedance - 1) == 0
    assert sp.Abs(complexe) == sp.Rational(1, 5)
    assert sp.simplify(sp.arg(complexe) + module_et_argument(impedance)[1]) == 0


def test_les_admittances_en_parallele_s_ajoutent() -> None:
    """(16.12.7), contre l'association (16.10.2) des impedances."""
    premiere = impedance_de_la_branche(3, 2, sp.oo, 2)
    seconde = impedance_de_la_branche(1, 0, 1, 2)
    somme = [
        a + b for a, b in zip(admittance(premiere), admittance(seconde), strict=True)
    ]
    assert tuple(somme) == admittance(en_parallele(premiere, seconde))


ZA, ZB, ZC = 3 + 4 * sp.I, sp.Integer(2), -5 * sp.I
QUADRIPOLE = (
    Branche("0", "1", 0),
    Branche("2", "0", 0),
    Branche("1", "m", ZA),
    Branche("m", "2", ZB),
    Branche("m", "0", ZC),
)
"""Figure 16.12.1 : les bornes 1-1' et 2-2' d'un reseau passif en T."""
ARBRE = (2, 3, 4)


def test_kirchhoff_en_complexes_par_la_methode_des_mailles() -> None:
    """(16.8.2) : les impedances a la place des resistances, rien d'autre.

    Une source `E` entre 1 et 1', un court-circuit entre 2 et 2' :
    `I_2 = E Z_c / (Z_a Z_b + Z_a Z_c + Z_b Z_c)`.
    """
    source = sp.Integer(10)
    alimente = (Branche("0", "1", 0, source), *QUADRIPOLE[1:])
    courants = resoudre_par_les_mailles(alimente, ARBRE)
    attendu = source * ZC / (ZA * ZB + ZA * ZC + ZB * ZC)
    assert sp.simplify(courants[1] - attendu) == 0


def test_les_admittances_de_transfert_sont_reciproques() -> None:
    """(16.12.8) et page 16-22 : `Y_12 = Y_21`, par le theoreme de reciprocite."""
    aller = conductance_de_transfert(QUADRIPOLE, ARBRE, source=0, mesure=1)
    retour = conductance_de_transfert(QUADRIPOLE, ARBRE, source=1, mesure=0)
    assert sp.simplify(aller - retour) == 0


def test_l_impedance_de_transfert_n_est_pas_l_inverse_de_l_admittance() -> None:
    """(16.12.9) : a bornes 2-2' ouvertes, `Z_12 = U_2 / I_1 = Z_c` pour un T."""
    admittance_12 = conductance_de_transfert(QUADRIPOLE, ARBRE, source=0, mesure=1)
    assert sp.simplify(ZC - 1 / admittance_12) != 0
