"""Planches d'electromagnetisme, chapitres IV.3 et IV.4.

Sources : `Electromagnetisme/SLIDES/Slides_04_Etats_variables_partie_3.pdf`
(etats quasi-stationnaires, planches 2 a 16) et
`Slides_04_Etats_variables_partie_4.pdf` (circuits a courants sinusoidaux,
planches 2 a 39), lues sur les pages rendues.

Les deux fascicules resument les chapitres 15 et 16 du syllabus : aucun
code neuf ici. Chaque relation encadree est rattachee au symbole qui la
porte, sur les nombres que les planches donnent : 6 000 et 30 km, -60 et
70 degres.
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
    courant_de_la_branche,
    impedance_de_la_branche,
    module_et_argument,
    reactance,
)
from polytech.electromagnetisme.domain.circuits.grandeur_sinusoidale import (
    GrandeurSinusoidale,
    RelationDePhase,
    dephasage,
    relation_de_phase,
    substitut_de_la_derivee,
)
from polytech.electromagnetisme.domain.circuits.periodiques import (
    puissance_apparente,
)
from polytech.electromagnetisme.domain.circuits.pertes_des_composants import (
    bobine_en_parallele,
    facteur_de_dissipation,
    impedance_d_une_resistance_reelle,
)
from polytech.electromagnetisme.domain.circuits.phaseurs import (
    puissance_complexe,
    valeur_instantanee,
)
from polytech.electromagnetisme.domain.circuits.puissances_sinusoidales import (
    Role,
    pertes_en_ligne,
    puissance_active,
    puissance_instantanee,
    puissance_reactive,
    role_et_nature,
)
from polytech.electromagnetisme.domain.circuits.sinusoidaux import (
    en_parallele,
    en_serie,
)
from polytech.electromagnetisme.domain.circuits.transformateur import (
    TransformateurParfait,
)
from polytech.electromagnetisme.domain.circuits.valeurs_efficaces import (
    valeur_efficace,
)
from polytech.electromagnetisme.domain.magnetostatique.cycle_dynamique import (
    pertes_magnetiques,
)
from polytech.electromagnetisme.domain.onde import (
    celerite_dans_le_vide,
    longueur_d_onde,
)
from polytech.electromagnetisme.domain.quasi_stationnaire import (
    rapport_a_la_longueur_d_onde,
)
from polytech.electromagnetisme.domain.symboles import TEMPS

OMEGA = sp.Symbol("omega", positive=True)


def test_la_condition_des_etats_quasi_stationnaires() -> None:
    """IV.3, planche 4 : `l` petit devant `lambda = v T` ; 6 000 km et 30 km."""
    celerite = celerite_dans_le_vide()
    assert longueur_d_onde(celerite, 50.0) == pytest.approx(6.0e6, rel=1e-3)
    assert longueur_d_onde(celerite, 1e4) == pytest.approx(30e3, rel=1e-3)
    assert rapport_a_la_longueur_d_onde(1.0, 50.0) < 1e-6


def test_l_equation_d_une_branche_complete() -> None:
    """IV.3, planche 11 : `u = e + R i + L di/dt + M di'/dt + q / C`, `i = dq/dt`."""
    fem, resistance, inductance, capacite, mutuelle = sp.symbols("e R L C M", real=True)
    charge, voisin = sp.Function("q")(TEMPS), sp.Function("i_p")(TEMPS)
    branche = BrancheFondamentale(fem, resistance, inductance, capacite)
    courant = sp.diff(charge, TEMPS)
    attendue = fem + resistance * courant + inductance * sp.diff(courant, TEMPS)
    attendue += mutuelle * sp.diff(voisin, TEMPS) + charge / capacite
    obtenue = branche.tension(charge, mutuelle=mutuelle, voisin=voisin)
    assert sp.simplify(obtenue - attendue) == 0


def test_la_tension_d_une_bobine_seule_est_l_di_sur_dt() -> None:
    """IV.3, planche 9 : `e = d(phi)/dt = N d(Phi)/dt = L di/dt`."""
    inductance = sp.Symbol("L", positive=True)
    charge = sp.Function("q")(TEMPS)
    seule = BrancheFondamentale(0, 0, inductance)
    assert seule.tension(charge) == inductance * sp.diff(charge, TEMPS, 2)


def test_les_courants_de_foucault_croissent_comme_le_carre_de_la_frequence() -> None:
    """IV.3, planches 15 et 16 : des pertes Joule, que l'on reduit en feuilletant."""
    assert pertes_magnetiques(0.0, 4e-4, 400.0) == pytest.approx(
        64 * pertes_magnetiques(0.0, 4e-4, 50.0)
    )


def test_elever_la_tension_reduit_les_pertes_en_ligne() -> None:
    """IV.4, planche 2 : `P = U I`, pertes `R I^2` ; dix fois `U`, cent fois moins."""
    basse = pertes_en_ligne(0.5, puissance=2300.0, tension=230.0, facteur=1.0)
    haute = pertes_en_ligne(0.5, puissance=2300.0, tension=2300.0, facteur=1.0)
    assert basse == pytest.approx(100 * haute)


def test_la_grandeur_sinusoidale_et_sa_valeur_efficace() -> None:
    """IV.4, planches 4 a 8 : `T = 2 pi / omega`, `f = 1 / T`, `X = x_m / racine(2)`."""
    grandeur = GrandeurSinusoidale(
        efficace=sp.Integer(3), phase=sp.pi / 6, pulsation=OMEGA
    )
    assert grandeur.frequence() == OMEGA / (2 * sp.pi)
    quadratique = valeur_efficace(grandeur.instantanee(), grandeur.periode())
    assert sp.simplify(quadratique - 3) == 0


def test_le_substitut_complexe_et_ses_proprietes() -> None:
    """IV.4, planche 10 : `|X| = X`, `arg X = theta_x`, derivee = `j omega`."""
    grandeur = GrandeurSinusoidale(
        efficace=sp.Integer(3), phase=sp.pi / 6, pulsation=OMEGA
    )
    substitut = grandeur.substitut()
    assert sp.Abs(substitut) == 3
    assert sp.arg(substitut) == sp.pi / 6
    assert sp.arg(sp.conjugate(substitut)) == -sp.pi / 6
    derive = valeur_instantanee(substitut_de_la_derivee(substitut, OMEGA), OMEGA)
    assert sp.simplify(derive - sp.diff(grandeur.instantanee(), TEMPS)) == 0


def test_le_dephasage_de_la_planche_11() -> None:
    """IV.4 : `theta_x = -60`, `theta_y = 70` degres ; quadrature, opposition."""
    degre = sp.pi / 180
    assert dephasage(70 * degre, sur=-60 * degre) == 130 * degre
    assert relation_de_phase(sp.pi / 2) is RelationDePhase.EN_QUADRATURE
    assert relation_de_phase(-sp.pi) is RelationDePhase.EN_OPPOSITION


def test_l_impedance_complexe_et_ses_associations() -> None:
    """IV.4, planche 15 : `Z = R + j X`, serie, parallele, admittance."""
    impedance = impedance_de_la_branche(3, 2, sp.Rational(1, 2), 2)
    assert impedance == 3 + 3 * sp.I
    assert reactance(2, sp.Rational(1, 2), 2) == 3
    assert module_et_argument(impedance) == (3 * sp.sqrt(2), sp.pi / 4)
    assert en_serie(impedance, sp.Integer(1)) == 4 + 3 * sp.I
    parallele = en_parallele(impedance, impedance)
    assert sp.simplify(parallele - impedance / 2) == 0
    conductance, susceptance = admittance(impedance)
    assert (conductance, susceptance) == (sp.Rational(1, 6), -sp.Rational(1, 6))


def test_les_circuits_de_base_des_planches_17_a_20() -> None:
    """IV.4 : `psi` vaut 0, `+pi/2`, `-pi/2`, puis reste entre les deux."""
    resistance = module_et_argument(impedance_de_la_branche(3, 0, sp.oo, 2))[1]
    bobine = module_et_argument(impedance_de_la_branche(0, 2, sp.oo, 2))[1]
    condensateur = module_et_argument(impedance_de_la_branche(0, 0, 1, 2))[1]
    assert (resistance, bobine, condensateur) == (0, sp.pi / 2, -sp.pi / 2)
    reelle = module_et_argument(impedance_de_la_branche(3, 2, sp.oo, 2))[1]
    assert 0 < reelle < sp.pi / 2


def test_avec_une_force_electromotrice_la_branche_suit_u_egale_e_plus_z_i() -> None:
    """IV.4, planche 21 : `U = E + Z I`."""
    impedance = impedance_de_la_branche(3, 2, sp.Rational(1, 2), 2)
    courant = courant_de_la_branche(sp.Integer(10), impedance, fem=4 * sp.I)
    assert sp.simplify(4 * sp.I + impedance * courant - 10) == 0


def test_les_puissances_des_planches_23_a_27() -> None:
    """IV.4 : `p = P + U I cos(2 omega t + 2 theta_u - phi)`, `S = U I* = P + j Q`."""
    tension, courant, ecart = 10, 2, sp.pi / 3
    active = puissance_active(tension, courant, ecart)
    reactive = puissance_reactive(tension, courant, ecart)
    assert (active, reactive) == (10, 10 * sp.sqrt(3))
    instantanee = puissance_instantanee(tension, courant, 0, ecart, OMEGA)
    # La puissance fluctuante culmine a `U I` au-dessus de la moyenne.
    crete = instantanee.subs(TEMPS, sp.pi / (6 * OMEGA))
    assert sp.simplify(crete - (active + tension * courant)) == 0
    complexe = puissance_complexe(sp.Integer(tension), courant * sp.exp(-sp.I * ecart))
    assert sp.expand_complex(complexe - (active + sp.I * reactive)) == 0
    assert puissance_apparente(10.0, 2.0) == pytest.approx(float(sp.Abs(complexe)))
    assert role_et_nature(float(active), float(reactive)) == (
        Role.RECEPTEUR,
        Nature.INDUCTIVE,
    )


def test_les_schemas_equivalents_des_planches_34_a_36() -> None:
    """IV.4 : `d = omega C_s R_s`, `tg(delta) = 1 / Q`, puis `L - C R^2`."""
    assert facteur_de_dissipation(100.0, 1e-4, 2.0) == pytest.approx(0.02)
    qualite = 100.0 * 0.5 / 2.0
    _, inductance = bobine_en_parallele(2.0, 0.5, qualite)
    assert inductance == pytest.approx(0.5, rel=2e-3)
    faible = impedance_d_une_resistance_reelle(1.0, 1e-6, 1e-12, 1e3)
    assert sp.im(faible) == pytest.approx(1e3 * (1e-6 - 1e-12), rel=1e-6)


def test_le_transformateur_parfait_des_planches_38_et_39() -> None:
    """IV.4 : `e_1 = N_1 dPhi/dt`, `e_2 = N_2 dPhi/dt`, d'ou `u_2 = (N_2 / N_1) u_1`."""
    transformateur = TransformateurParfait(primaire=500, secondaire=50)
    flux = sp.sin(OMEGA * TEMPS)
    primaire = 500 * sp.diff(flux, TEMPS)
    secondaire = 50 * sp.diff(flux, TEMPS)
    assert sp.simplify(transformateur.tension_secondaire(primaire) - secondaire) == 0
