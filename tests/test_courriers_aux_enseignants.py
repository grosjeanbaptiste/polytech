"""Valeurs des courriers aux enseignants, figees par la bibliotheque.

Complete les tests des syllabus : pour chaque puce d'un courrier, la valeur
proposee (tolerance serree) et la valeur citee, comparee et declaree
differente de celle que la bibliotheque calcule.
"""

from __future__ import annotations

import cmath
import itertools
import math

import numpy as np
import pytest
import sympy as sp

from polytech.chemistry.domain.atomes import RAYON_DE_BOHR, Atome
from polytech.electromagnetisme.application import cote_2016_enonce as en
from polytech.electromagnetisme.application import cote_2016_magnetique as ma
from polytech.electromagnetisme.application import cote_2016_sinusoidal as si
from polytech.electromagnetisme.application import cote_2023_mons_bobines as bo
from polytech.electromagnetisme.application import cote_2024_mons_commutation as co
from polytech.electromagnetisme.application import travail_statique_12 as tr
from polytech.electromagnetisme.application.enonces_seance_32 import (
    PRIMAIRE_DU_SIX,
    circuit_du_six,
    courant_secondaire_du_six,
)
from polytech.electromagnetisme.application.symboles_seance_32 import OMEGA_DU_SIX
from polytech.electromagnetisme.domain.charges import PARTICULES, charge_d_un_systeme
from polytech.electromagnetisme.domain.circuits.associations import (
    courant_de_maille,
    tension_aux_bornes_de_la_seconde,
    thevenin_de_la_maille,
)
from polytech.electromagnetisme.domain.circuits.parasites import (
    courant_rc_avec_inductance_parasite,
)
from polytech.electromagnetisme.domain.circuits.phaseurs import en_regime_sinusoidal
from polytech.electromagnetisme.domain.constantes import (
    CHARGE_ELEMENTAIRE,
    PERMITTIVITE_DU_VIDE,
)
from polytech.electromagnetisme.domain.electrostatique.conducteurs import (
    charge_surfacique_depuis_le_champ,
)
from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.electrostatique.energie import (
    energie_de_la_sphere,
)
from polytech.electromagnetisme.domain.magnetostatique.fil_et_spire import (
    SpireDevantUnFil,
    force_cote_par_cote,
)
from polytech.electromagnetisme.domain.magnetostatique.materiaux_pour_aimants import (
    borne_du_produit_d_energie,
)
from polytech.electromagnetisme.domain.matiere import Echantillon
from polytech.electromagnetisme.domain.moyennes_macroscopiques import (
    charge_volumique_d_une_sphere,
)
from polytech.electromagnetisme.domain.symboles import TEMPS as TEMPS_EM
from polytech.fluides.domain.fondamentales.bernoulli import GRAVITE_EXACTE
from polytech.fluides.domain.fondamentales.clepsydre import (
    baisse_du_niveau,
    coefficient_du_profil,
    rayon_de_l_orifice,
)
from polytech.maths.domain.algebre.complexes_exacts import forme_polaire
from polytech.maths.domain.algebre.exponentielle import par_jordan
from polytech.maths.domain.algebre.spectre import Spectre
from polytech.maths.domain.algebre.triangularisation import (
    produit_de_passage,
    reconstruit_par,
    semblable_par_inverse_donnee,
)
from polytech.maths.domain.differentielles.familles import (
    pentes_de_la_famille,
    valeurs_de_la_constante,
)
from polytech.maths.domain.differentielles.formes_sinusoidales import (
    en_cosinus,
    en_sinus,
)
from polytech.maths.domain.differentielles.predateur_proie import PredateurProie
from polytech.maths.domain.differentielles.premier_ordre import INCONNUE
from polytech.maths.domain.differentielles.premier_ordre import VARIABLE as XD
from polytech.maths.domain.edp.caracterisation import est_solution_de_l_edp
from polytech.maths.domain.edp.euler import EquationDEuler
from polytech.maths.domain.fourier.dirac import dirac_d_une_fonction
from polytech.maths.domain.fourier.dispersion import ecart_type
from polytech.maths.domain.fourier.distributions import (
    FONCTION_TEST,
    appliquer_le_dirac,
    est_a_decroissance_rapide,
    transformee_de_la_constante,
)
from polytech.maths.domain.fourier.proprietes import de_la_modulation
from polytech.maths.domain.fourier.transformee import PULSATION as W
from polytech.maths.domain.fourier.transformee import (
    integrale_sur_la_droite,
    transformee_de_fourier,
    transformee_inverse,
)
from polytech.maths.domain.integration.controle import ecart_de_derivation
from polytech.maths.domain.integration.fractions import FractionRationnelle
from polytech.maths.domain.integration.primitives import Primitive
from polytech.maths.domain.laplace.theoremes import valeur_initiale
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T
from polytech.maths.domain.symboles import VARIABLE
from polytech.mecanique.domain.gravitation import force_gravitationnelle
from polytech.optimisation.application import modele_gare as ga
from polytech.optimisation.application import modele_gare_categories as ca
from polytech.optimisation.application import modele_production as pr
from polytech.signaux.application import enonce_lab_01 as lab
from polytech.signaux.application import resolus_convolutions as cv
from polytech.signaux.application import resolus_puissances as pu
from polytech.signaux.domain.modeles import coefficients, transmittance_d_une_equation
from polytech.signaux.domain.poles_numeriques import poles_et_zeros
from polytech.signaux.domain.reponse_indicielle import (
    pic_de_resonance_du_second_ordre,
)

X = VARIABLE
CUIVRE = Atome("Cu", numero_atomique=29, nombre_de_masse=63)


def _cuivre() -> Echantillon:
    return Echantillon(
        CUIVRE, masse_volumique=8960, masse_molaire=63.546e-3, volume=1e-9
    )


def test_m1_les_deux_dephasages_valent_pi_sur_deux_et_pas_moins_pi_sur_deux() -> None:
    assert en_sinus(1, 1) == (sp.sqrt(2), sp.pi / 4)
    assert en_cosinus(1, 1) == (sp.sqrt(2), -sp.pi / 4)
    _, phi = en_sinus(3, 4)
    _, psi = en_cosinus(3, 4)
    assert sp.simplify(sp.expand_trig(sp.sin(phi - psi))) == 1
    assert sp.simplify(sp.expand_trig(sp.sin(phi - (psi - sp.pi / 2)))) != 1


def test_m13_le_coefficient_croise_n_est_pas_2_ab_moins_h2_sur_b() -> None:
    premiere, seconde = EquationDEuler(1, 3, 2).pentes()
    _, c2, _ = EquationDEuler(1, 3, 2).coefficients_transformes(1, premiere, 1, seconde)
    assert sp.simplify(c2) == -14  # 4 (ab - h^2) / b avec a=1, h=3, b=2
    assert sp.simplify(c2) != -7  # 2 (ab - h^2) / b


def _egales_en_w(obtenue: sp.Expr, attendue: sp.Expr) -> bool:
    # sympy laisse des |w| qu'il ne simplifie pas : comparaison en trois pulsations
    ecart = sp.simplify(obtenue - attendue)
    valeurs = (1, sp.Rational(-7, 3), sp.Rational(5, 2))
    return all(sp.simplify(ecart.subs(W, v)) == 0 for v in valeurs)


def test_m16_la_modulation_donne_cos_et_sin_sans_j() -> None:
    f = sp.exp(-(XD**2))
    image = transformee_de_fourier(f)
    directe = transformee_de_fourier(f * sp.exp(3 * sp.I * XD))
    assert sp.simplify(de_la_modulation(image, 3) - directe) == 0
    cosinus = transformee_de_fourier(f * sp.cos(3 * XD))
    somme = (de_la_modulation(image, 3) + de_la_modulation(image, -3)) / 2
    assert _egales_en_w(cosinus, somme)
    sinus = transformee_de_fourier(f * sp.sin(3 * XD))
    difference = (de_la_modulation(image, 3) - de_la_modulation(image, -3)) / (2 * sp.I)
    assert _egales_en_w(sinus, difference)


def test_e1_les_charges_du_cuivre_proposees() -> None:
    echantillon = _cuivre()
    assert round(echantillon.charge_positive, 1) == 394.5
    assert round(echantillon.charge_mobile(1), 2) == -13.60
    assert round(echantillon.charge_mobile(2), 2) == -27.21
    assert echantillon.charge_mobile(2) == pytest.approx(
        2 * echantillon.charge_mobile(1), rel=1e-12
    )
    assert round(echantillon.charge_mobile(2), 2) != -27.03
    assert round(echantillon.charge_positive, 1) not in (392.0, 394.0)


def test_e3_une_fonction_a_pole_ne_porte_pas_de_pic() -> None:
    assert dirac_d_une_fonction(1 / X) == {}
    assert dirac_d_une_fonction(X**2 - 1) == {
        sp.Integer(-1): sp.Rational(1, 2),
        sp.Integer(1): sp.Rational(1, 2),
    }


def test_e5_e6_la_gravitation_et_les_poids_de_l_exemple_2_20_1() -> None:
    # 0,9e-30 kg est la masse de l'enonce (donnee imprimee, pas un calcul).
    m_e = 0.9e-30
    imprime = force_gravitationnelle(m_e, 1836 * m_e, RAYON_DE_BOHR)
    assert round(imprime * 1e48, 1) == 35.4
    ecrit = force_gravitationnelle(0.9e-19, 1846 * 0.9e-19, RAYON_DE_BOHR)
    assert ecrit / imprime > 1e22
    masses = PARTICULES["electron"].masse * PARTICULES["proton"].masse
    assert masses / PARTICULES["electron"].masse ** 2 == pytest.approx(
        1836.645, abs=1e-3
    )
    proton = PARTICULES["proton"].masse
    assert round(proton * 10 * 1e27, 1) == 16.7
    assert round(proton * 9.81 * 1e27, 1) == 16.4


def test_e8_la_charge_surfacique_de_la_terre_est_un_nanocoulomb() -> None:
    sigma = float(
        charge_surfacique_depuis_le_champ(100).subs(PERMITTIVITE, PERMITTIVITE_DU_VIDE)
    )
    assert round(sigma * 1e9, 3) == 0.885
    assert sigma / 0.9 < 1.1e-9


def test_e9_le_produit_d_energie_des_ferrites_imprimes_plafonne_a_vingt() -> None:
    assert borne_du_produit_d_energie(1.0, 20.0) == pytest.approx(20.0, rel=1e-12)
    assert borne_du_produit_d_energie(1.0, 10.0) == pytest.approx(10.0, rel=1e-12)


def test_e18_le_dephasage_du_courant_sur_la_tension() -> None:
    tension, courant = si.tension_de_la_branche(), si.courant_de_la_branche()
    arg_u = math.degrees(cmath.phase(tension))
    arg_i = math.degrees(cmath.phase(courant))
    assert arg_u == pytest.approx(1.9554, abs=5e-4)
    assert arg_i == pytest.approx(-1.6399, abs=5e-4)
    assert si.dephasage_du_courant() == pytest.approx(-3.5953, abs=5e-4)
    assert arg_i - arg_u == pytest.approx(si.dephasage_du_courant(), abs=1e-9)
    assert arg_i + arg_u == pytest.approx(0.3155, abs=5e-4)  # avec -(-1,95)


def test_e21_le_courant_avant_l_ouverture_passe_par_r12_et_pas_par_r11() -> None:
    avant_i1 = co.avant()[0]
    assert avant_i1 == -co.E1 / co.R12
    assert avant_i1 != -co.E1 / co.R11


def test_e24_les_cotes_haut_et_bas_ont_la_force_de_module_donne() -> None:
    p, i1, i2 = sp.symbols("mu_0 I_1 I_2", positive=True)
    a, b, d = sp.symbols("a b d", positive=True)
    forces = force_cote_par_cote(SpireDevantUnFil(a, b, d), i1, i2, p)
    attendu = p * i1 * i2 / (2 * sp.pi) * sp.log((a + d) / d)
    assert sp.simplify(forces["haut"][1] ** 2 - attendu**2) == 0
    assert sp.simplify(forces["bas"][1] ** 2 - attendu**2) == 0


def test_s4_le_rapport_signal_sur_bruit_initial_et_apres_attenuation() -> None:
    assert pu.rapport_signal_sur_bruit(gain_du_bruit=1) == pytest.approx(
        -3.0103, abs=5e-5
    )
    assert pu.rapport_signal_sur_bruit(gain_du_bruit=0.01) == pytest.approx(
        36.9897, abs=5e-5
    )
    assert pu.rapport_signal_sur_bruit(gain_du_bruit=100) == pytest.approx(
        -43.0103, abs=5e-5
    )


def test_s9_le_pic_de_resonance_vaut_17_5_pour_k_10_xi_0_3() -> None:
    assert 10 * pic_de_resonance_du_second_ordre(0.3) == pytest.approx(
        17.4714, abs=5e-4
    )


def test_lab1_les_poles_de_la_capsule_et_le_critere_de_routh() -> None:
    numerateur, denominateur = coefficients(
        transmittance_d_une_equation(lab.CINQ_SORTIE, lab.CINQ_ENTREE)
    )
    poles, _, _ = poles_et_zeros(numerateur, denominateur)
    classes = sorted(poles, key=lambda z: z.real)
    assert classes[0].real == pytest.approx(-1.4164, abs=5e-5)
    assert classes[1].real == pytest.approx(0.2082, abs=5e-5)
    assert abs(classes[1].imag) == pytest.approx(3.2475, abs=5e-5)
    a3, a2, a1, a0 = lab.CINQ_SORTIE
    assert a2 * a1 - a3 * a0 == -5
    assert int(np.sum(poles.real > 0)) == 2


def test_o4_la_gare_a_besoin_de_trois_employes_de_0_h_a_4_h() -> None:
    assert ga.BESOINS[0] == 3


def test_e2_la_charge_volumique_du_noyau_d_uranium() -> None:
    eta = charge_volumique_d_une_sphere(charge_d_un_systeme(92), 1e-15)
    assert round(eta / 1e27, 2) == 3.52
    assert eta / 35e27 == pytest.approx(0.1005, abs=5e-4)


def test_m11_proies_et_predateurs_seuls_sont_exponentiels_en_t() -> None:
    a, alpha, c, gamma = sp.symbols("a alpha c gamma", positive=True)
    proies, predateurs = PredateurProie(a, alpha, c, gamma).evolution_isolee()
    x, y = PredateurProie.POPULATIONS
    assert proies == sp.Symbol("C", real=True) * sp.exp(a * T)
    assert predateurs == sp.Symbol("C", real=True) * sp.exp(-c * T)
    assert not proies.has(x, y)
    assert not predateurs.has(x, y)  # e^(-cy) ne serait pas une fonction de t


def test_m32_la_primitive_imprimee_de_la_planche_39_n_en_est_pas_une() -> None:
    integrande = 1 / (X**2 - 4)
    imprimee = (X - 2) / (X + 2) / 4
    assert not Primitive(integrande, imprimee).verifiee
    assert (
        abs(ecart_de_derivation(imprimee, integrande, [3]) - sp.Rational(4, 25)) < 1e-25
    )
    corrigee = FractionRationnelle(sp.Integer(1), X**2 - 4).primitive()
    assert Primitive(integrande, corrigee).verifiee
    assert Primitive(integrande, sp.log((X - 2) / (X + 2)) / 4).verifiee


def test_m33_les_exponentielles_de_ii_7_3_a_sont_echangees() -> None:
    j = sp.I
    communes = -sp.exp(-T) / 2 + sp.Rational(2, 5) * sp.exp(-2 * T)
    imprimee = (
        communes
        + sp.exp(-j * T) / (2 * (1 + 3 * j))
        + sp.exp(j * T) / (2 * (1 - 3 * j))
    )
    corrigee = (
        communes
        + sp.exp(j * T) / (2 * (1 + 3 * j))
        + sp.exp(-j * T) / (2 * (1 - 3 * j))
    )

    def pente(original: sp.Expr) -> sp.Expr:
        image = Transformee.de(original).image
        derivee = Transformee.de_la_derivee(image, valeur_initiale(image))
        return sp.simplify(valeur_initiale(derivee))

    assert pente(imprimee) == sp.Rational(-3, 5)
    assert pente(corrigee) == 0
    # le residu en p = j de p/((p^2+1)(p+1)(p+2)) multiplie e^(jt)
    reponse = Transformee(
        sp.Integer(0), P / ((P**2 + 1) * (P**2 + 3 * P + 2))
    ).inverse()
    reponse = reponse.subs(sp.Heaviside(T), 1)
    assert sp.simplify(sp.re(sp.expand_complex(corrigee)) - reponse) == 0
    assert sp.simplify(sp.re(sp.expand_complex(imprimee)) - reponse) != 0


def test_e23_l_exercice_6_rend_163_375_degres_et_ni_163_71_ni_163_38() -> None:
    primaire = en_regime_sinusoidal(
        circuit_du_six().courant(PRIMAIRE_DU_SIX), OMEGA_DU_SIX
    )
    secondaire = en_regime_sinusoidal(courant_secondaire_du_six(), OMEGA_DU_SIX)
    rapport = complex(sp.N(primaire / secondaire, 30))
    degres = math.degrees(cmath.phase(rapport))
    assert degres == pytest.approx(-163.375, abs=5e-4)
    assert abs(degres - (-163.71)) > 0.3
    assert abs(degres - (-163.38)) > 4e-3


def test_e26_le_travail_a_potentiels_constants_s_ecrit_en_u_et_pas_en_q() -> None:
    cas = tr.a_potentiels_constants()
    assert cas.travail.has(tr.TENSION)
    assert not cas.travail.has(tr.CHARGE)
    assert not cas.initiale.has(tr.CHARGE)
    assert not cas.finale.has(tr.CHARGE)


def test_o7_les_demandes_cumulees_de_la_planche_69() -> None:
    assert tuple(itertools.accumulate(pr.DEMANDES[:-1])) == (5, 11, 19, 25)
    assert tuple(itertools.accumulate(pr.ALTERNATIVE[:-1])) == (7, 12, 15, 24)


def test_m3_la_constante_du_complement_1_a_est_k_carre() -> None:
    k, c = sp.Symbol("K", real=True), sp.Symbol("c", real=True)
    famille = INCONNUE - (sp.sqrt(XD) + k) ** 2
    y = (sp.sqrt(XD) + k) ** 2
    for pente in pentes_de_la_famille(famille, k):
        assert sp.simplify((XD * pente**2 - INCONNUE).subs(INCONNUE, y)) == 0
    relation = (XD - INCONNUE) ** 2 - 2 * c * (XD + INCONNUE) + c**2
    assert valeurs_de_la_constante(relation, y, c) == (k**2,)
    assert valeurs_de_la_constante(relation, y, c) != (k,)


def test_m12_f_de_deux_variables_donne_2y_pour_u_xx_moins_u_yy() -> None:
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")(x, y)
    f, g = sp.Function("f"), sp.Function("g")
    ondes = sp.diff(u, x, 2) - sp.diff(u, y, 2)
    assert est_solution_de_l_edp(ondes, u, f(x + y) + g(x - y))
    assert not est_solution_de_l_edp(ondes, u, f(x, y) + g(x - y))
    residu = ondes.subs(u, x**2 * y + g(x - y)).doit()
    assert sp.simplify(residu) == 2 * y


def test_m15_m16_m19_les_distributions_du_chapitre_10() -> None:
    a = sp.Symbol("a", positive=True)
    assert integrale_sur_la_droite(2 * sp.sin(W * a) / W, W) == 2 * sp.pi
    assert transformee_de_la_constante(1) == 2 * sp.pi * sp.DiracDelta(W)
    t = sp.Symbol("t", real=True)
    assert appliquer_le_dirac(0, sp.exp(sp.I * XD * t)) == 1
    assert FONCTION_TEST.subs(XD, 0) == sp.exp(-1)
    assert FONCTION_TEST.subs(XD, 2) == 0
    assert FONCTION_TEST.subs(XD, -2) == 0
    assert sp.exp(1 / (XD**2 - 1)).subs(XD, -2) == sp.exp(sp.Rational(1, 3))


def test_m18_la_constante_est_bornee_pour_n_0_et_pas_pour_n_1() -> None:
    assert est_a_decroissance_rapide(sp.Integer(1), 0, 0)
    assert not est_a_decroissance_rapide(sp.Integer(1), 1, 0)


def test_m20_le_produit_3_4i_par_2_5i_n_est_pas_26_moins_7i() -> None:
    def theta(z: sp.Expr) -> float:
        return float(sp.N(forme_polaire(z)[1], 30))

    mod_p, _ = forme_polaire(3 + 4 * sp.I)
    # arguments : un produit somme ceux de ses facteurs
    assert theta(26 - 7 * sp.I) == pytest.approx(
        theta(3 + 4 * sp.I) + theta(2 - 5 * sp.I), abs=1e-12
    )
    assert theta(-14 + 23 * sp.I) == pytest.approx(
        theta(3 + 4 * sp.I) + theta(2 + 5 * sp.I), abs=1e-12
    )
    assert theta(26 - 7 * sp.I) != pytest.approx(
        theta(3 + 4 * sp.I) + theta(2 + 5 * sp.I), abs=1e-6
    )
    assert mod_p == 5


def test_m8_des_espaces_propres_distincts_ne_sont_pas_orthogonaux() -> None:
    jordan = sp.Matrix([[1, 0, 1], [2, -2, -1], [0, 0, 1]])
    couples = dict(Spectre(jordan).couples_propres())
    v1, v2 = couples[-2], couples[1]
    assert v1.dot(v2) != 0
    symetrique = dict(Spectre(sp.Matrix([[2, 1], [1, 2]])).couples_propres())
    premier, second = symetrique.values()
    assert premier.dot(second) == 0


def test_m5_le_rayon_de_l_horloge_est_1_sur_950_et_pas_1_sur_5400() -> None:
    b = coefficient_du_profil(2, sp.Rational(1, 2))
    juste = rayon_de_l_orifice(b, sp.Rational(1, 10) / 3600)
    assert round(float(juste) * 1e3, 4) == 1.0529
    assert round(1 / float(juste)) == 950
    r = sp.Symbol("r", positive=True)
    k = sp.pi * sp.Rational(1, 5400) ** 2 * sp.sqrt(2 * GRAVITE_EXACTE)
    par_heure = -baisse_du_niveau(b**2 * r**4, r, k) * 3600
    assert round(float(par_heure) * 1e3, 1) == 3.1


def test_e7_l_energie_du_noyau_d_uranium_est_812_5_mev_et_pas_813_5() -> None:
    joules = float(
        energie_de_la_sphere(92 * CHARGE_ELEMENTAIRE, 9e-15).subs(
            PERMITTIVITE, PERMITTIVITE_DU_VIDE
        )
    )
    assert round(joules * 1e10, 1) == 1.3
    mev = joules / CHARGE_ELEMENTAIRE / 1e6
    assert round(mev, 1) == 812.5
    assert round(mev, 1) != 813.5


def test_o3_o4_la_gare_a_horaires_coupes() -> None:
    assert ca.cout_d_un_horaire_coupe(2, 4) == sp.Rational(132, 10)
    assert ca.COUT_IMPRIME[(2, 4)] == sp.Rational(133, 10)
    assert ca.horaires_de_la_periode(1) == ((1, 3), (1, 4), (1, 5))
    assert ca.LIGNE_IMPRIMEE_DE_LA_PERIODE_1 == ((2, 4), (2, 5), (2, 6))
    assert ca.presents(ca.CONTRE_EXEMPLE)[0] == 0
    assert ga.BESOINS[0] == 3
    assert float(ca.cout(ca.CONTRE_EXEMPLE)) == pytest.approx(249.6, abs=1e-9)
    assert float(ca.optimum()[0]) == pytest.approx(266.8, abs=1e-9)


def test_e20_la_phase_du_secondaire_est_126_77_degres() -> None:
    amplitude, phase = bo.sinusoide()
    assert round(amplitude, 2) == 69.82
    assert round(phase, 4) == -2.2125
    degres = math.degrees(-phase)
    assert round(degres, 2) == 126.77
    assert abs(degres - 126.6) > 0.1


def test_e14_le_courant_rc_a_t_sur_t_9_vaut_0_41_et_pas_exp_moins_90() -> None:
    reel = courant_rc_avec_inductance_parasite(1, 1, sp.Integer(5), 1)
    en_neuf = float(reel.subs(TEMPS_EM, 9).rewrite(sp.exp))
    assert round(en_neuf, 2) == 0.41
    assert en_neuf > 1e30 * math.exp(-90)


def test_m14_l_inverse_au_signe_moins_rend_f_de_moins_t() -> None:
    decalee = sp.exp(-((XD - 1) ** 2))
    image = transformee_de_fourier(decalee)
    assert (
        sp.simplify(transformee_inverse(image, signe=-1) - decalee.subs(XD, -XD)) == 0
    )
    assert sp.simplify(transformee_inverse(image) - decalee) == 0
    assert sp.simplify(transformee_inverse(image, signe=-1) - decalee) != 0


def test_m17_les_dispersions_de_la_gaussienne_sont_des_ecarts_types() -> None:
    a = sp.Symbol("a", positive=True)
    gauss = sp.exp(-((a * XD) ** 2))
    assert ecart_type(gauss, XD) == 1 / (a * sp.sqrt(2))
    assert (
        sp.simplify(ecart_type(transformee_de_fourier(gauss), W) - a * sp.sqrt(2)) == 0
    )
    assert ecart_type(gauss, XD) ** 2 == 1 / (2 * a**2)


def test_m9_m26_les_inverses_imprimes_des_pages_133_et_165() -> None:
    r = sp.Rational
    jordan = sp.Matrix([[1, 0, 1], [2, -2, -1], [0, 0, 1]])
    passage = sp.Matrix([[0, 3, 1], [2, 2, -1], [0, 0, 3]]) / 2
    blocs = sp.Matrix([[-2, 0, 0], [0, 1, 1], [0, 0, 1]])
    imprimee = sp.Matrix([[-6, 5, 5], [6, 0, -2], [0, 0, 6]]) / 9
    assert produit_de_passage(passage, imprimee) == sp.diag(1, r(5, 9), 1)
    assert reconstruit_par(passage, blocs, imprimee) != jordan
    fausse = reconstruit_par(passage, par_jordan(blocs), imprimee)
    assert sp.simplify(fausse[1, 1] - r(5, 9) * sp.exp(-2)) == 0
    a = sp.Matrix([[5, 10, -7], [-3, -5, 3], [-1, 2, -1]])
    p = sp.Matrix([[1, -26, 0], [0, 15, 0], [1, 14, 1]])
    p_inv = sp.Matrix([[1, r(26, 15), 0], [0, r(1, 15), 0], [-1, -r(40, 15), 1]])
    t = semblable_par_inverse_donnee(a, p, p_inv)
    assert t == sp.Matrix([[-2, 0, -r(9, 5)], [0, 3, r(1, 5)], [0, 0, -2]])
    assert t[1, 2] != 0


def test_e25_le_lapsus_de_thevenin_20_pour_10() -> None:
    courant = courant_de_maille(20.0, 10.0, 1.0, 5.0, 1.0)
    juste = tension_aux_bornes_de_la_seconde(10.0, 5.0, courant, 1.0)
    lapsus = tension_aux_bornes_de_la_seconde(20.0, 5.0, courant, 1.0)
    assert courant == pytest.approx(5 / 6, abs=1e-12)
    assert juste == pytest.approx(115 / 6, abs=1e-12)
    assert round(juste, 2) == 19.17
    assert round(lapsus, 2) == 29.17
    thevenin = thevenin_de_la_maille(20.0, 10.0, 1.0, 5.0, 1.0)
    assert thevenin.tension_a_vide == pytest.approx(juste, abs=1e-12)
    assert thevenin.resistance == pytest.approx(5 / 6, abs=1e-12)


def test_l11_la_donnee_de_1_h_vaut_16_fois_n_phi_sur_i() -> None:
    apparente = ma.inductance_apparente()
    assert round(apparente * 1e3, 1) == 61.5
    assert en.INDUCTANCE_PROPRE / apparente == pytest.approx(16.25, abs=1e-9)
    assert round(ma.courant_des_bobines(), 3) == 0.325


def test_s5_la_sortie_de_l_echelon_est_t_moins_1_et_pas_t() -> None:
    assert [cv.sortie_de_l_echelon(t) for t in (1, 2, sp.Rational(5, 2), 3)] == [
        0,
        1,
        sp.Rational(3, 2),
        2,
    ]
    assert cv.saut_si_le_milieu_valait_t() == (1, -1)
