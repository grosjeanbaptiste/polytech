"""Le syllabus d'electronique, chapitre 2 : ce qu'il ajoute aux planches.

Source : `Syllabus_Electronics_and_Instrumentation.pdf`, dans
`ElectroniqueInstrumentation/SYLLABUS/`, sections 2.8.1, 2.8.5, 2.11.6 et
2.12.3 (pages 20 a 27 et 51 a 62), lues sur leur couche de texte ; la
page 26 sur son rendu.

Les oracles : l'etude du conducteur cylindrique (equations (2.39) a
(2.41), figures 2.5 et 2.6), l'exemple 2-7 de decomposition a deux poles
doubles, l'exemple 2-10 de l'emetteur commun degenere, l'exemple 2-13 du
circuit RLC charge.

Un ecart, lu sur la page rendue : (2.45) ecrit `2 A sin(w t - pi l /
lambda) cos(pi l / lambda)` ; l'identite que la page invoque donne `2 A
sin(-pi l / lambda) cos(w t - pi l / lambda)`. A `l = lambda / 20`,
l'ecart a donc pour amplitude `2 sin(pi / 20) = 0,31` et non « 0,16 » :
l'erreur est bornee par 31 %, pas par 16 %.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.conducteur_cylindrique import (
    champ_electrique_uniforme,
    champ_magnetique_interne,
    rapport_conduction_deplacement,
)
from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE
from polytech.electromagnetisme.domain.effet_pelliculaire import (
    CONDUCTIVITE_DU_CUIVRE,
    epaisseur_de_peau,
)
from polytech.electronique.domain.delai_de_propagation import (
    amplitude_de_l_ecart,
    pire_ecart_sur_une_periode,
    retard_de_phase_en_degres,
)
from polytech.electronique.domain.laplace import (
    decomposition_en_fractions_simples,
    reponse_impulsionnelle,
)
from polytech.electronique.domain.netliste import (
    Admittance,
    SourceCommandee,
    SourceDeCourant,
    SourceDeTension,
    resoudre,
)

S = sp.Symbol("s")
T = sp.Symbol("t", positive=True)
CUIVRE = 5e7
"""Page 20 : « higher than 5 10^7 S/m at room temperature »."""


def test_le_courant_de_deplacement_est_negligeable_dans_un_conducteur() -> None:
    """Page 21 : `sigma / (eps_0 w) > 5e18 / w` ; 0,1 % vers 5e15 rad/s, 800 THz."""
    assert CONDUCTIVITE_DU_CUIVRE > CUIVRE
    rapport = rapport_conduction_deplacement(CUIVRE, 1.0, permittivite=8.8e-12)
    assert rapport > 5e18
    pulsation = 5e15
    seuil = rapport_conduction_deplacement(CUIVRE, pulsation, permittivite=8.8e-12)
    assert 1 / seuil == pytest.approx(1e-3, rel=0.15)
    assert pulsation / (2 * math.pi) == pytest.approx(8e14, rel=0.01)
    with pytest.raises(ValueError, match="pulsation nulle ou negative"):
        rapport_conduction_deplacement(CUIVRE, 0.0)
    with pytest.raises(ValueError, match="permittivite nulle ou negative"):
        rapport_conduction_deplacement(CUIVRE, 1.0, permittivite=0.0)


def test_un_courant_constant_se_repartit_uniformement() -> None:
    """Page 22 : `E_0 = I / (sigma pi a^2)`, et (2.41) `B = mu_0 sigma E_0 r / 2`."""
    courant, sigma, rayon, r = sp.symbols("I sigma a r", positive=True)
    champ = champ_electrique_uniforme(courant, sigma, rayon)
    assert sp.simplify(sigma * champ * sp.pi * rayon**2 - courant) == 0
    induction = champ_magnetique_interne(r, champ, sigma)
    mu = sp.Float(PERMEABILITE_DU_VIDE)
    ampere = 2 * sp.pi * r * induction - mu * sigma * champ * sp.pi * r**2
    assert sp.simplify(ampere) == 0
    rotationnel = sp.diff(r * induction, r) / r
    assert sp.simplify(rotationnel - mu * sigma * champ) == 0
    au_bord = induction.subs(r, rayon)
    assert sp.simplify(au_bord - mu * courant / (2 * sp.pi * rayon)) == 0
    with pytest.raises(ValueError, match="conductivite ou rayon nul"):
        champ_electrique_uniforme(courant, sigma, 0)


def test_un_courant_variable_ne_peut_pas_rester_uniforme() -> None:
    """Page 23 : uniforme, il laisse `w mu_0 sigma E_0 h r^2 cos(w t) / 4` a Faraday."""
    sigma, e0, r, h, w, t = sp.symbols("sigma E_0 r h omega t", positive=True)
    induction = champ_magnetique_interne(r, e0 * sp.sin(w * t), sigma)
    flux = sp.integrate(induction, (r, 0, r)) * h
    induite = -sp.diff(flux, t)
    mu = sp.Float(PERMEABILITE_DU_VIDE)
    attendue = -w * mu * sigma * e0 * h * r**2 * sp.cos(w * t) / 4
    assert sp.simplify(induite - attendue) == 0
    circulation = h * e0 * sp.sin(w * t) - h * e0 * sp.sin(w * t)
    assert circulation == 0
    assert sp.simplify(induite) != 0


def test_l_effet_de_peau_des_figures_2_5_et_2_6() -> None:
    """Page 24 : fil de 0,65 mm, uniforme a 50 Hz, 25 um utiles a 100 MHz."""
    rayon_fin, rayon_gros = 0.325e-3, 5e-3
    assert epaisseur_de_peau(50.0, CUIVRE) > 3 * rayon_fin
    assert epaisseur_de_peau(50.0, CUIVRE) > rayon_gros
    a_cent_kilohertz = epaisseur_de_peau(1e5, CUIVRE)
    assert rayon_fin / 2 < a_cent_kilohertz < rayon_fin
    assert a_cent_kilohertz < rayon_gros / 20
    a_cent_megahertz = epaisseur_de_peau(1e8, CUIVRE)
    assert 3 * a_cent_megahertz < 25e-6 < 4 * a_cent_megahertz
    grandit = epaisseur_de_peau(1e8, CUIVRE) / epaisseur_de_peau(4e8, CUIVRE)
    assert grandit == pytest.approx(2.0)


def test_l_ecart_du_au_delai_de_propagation() -> None:
    """(2.45) : `sin(w t - 2 pi x) - sin(w t)`, d'amplitude `2 sin(pi x)`."""
    rapport = sp.Symbol("x", positive=True)
    phase = sp.Symbol("theta", real=True)
    ecart = sp.sin(phase - 2 * sp.pi * rapport) - sp.sin(phase)
    identite = 2 * sp.sin(-sp.pi * rapport) * sp.cos(phase - sp.pi * rapport)
    assert sp.simplify(ecart - identite) == 0
    imprimee = 2 * sp.sin(phase - sp.pi * rapport) * sp.cos(sp.pi * rapport)
    assert sp.simplify(ecart - imprimee) != 0
    attendue = 2 * math.sin(math.pi / 20)
    assert amplitude_de_l_ecart(1.0, 20.0) == pytest.approx(attendue)
    with pytest.raises(ValueError, match="longueur d'onde nulle ou negative"):
        amplitude_de_l_ecart(1.0, 0.0)
    with pytest.raises(ValueError, match="longueur de ligne negative"):
        amplitude_de_l_ecart(-1.0, 20.0)


def test_a_un_vingtieme_de_longueur_d_onde_l_erreur_atteint_31_pour_cent() -> None:
    """(2.46) imprime « -0.16 cos(w t - 9 deg) » et « +-16 % » : c'est 0,31."""
    amplitude = amplitude_de_l_ecart(1.0, 20.0)
    assert amplitude == pytest.approx(0.313, abs=5e-4)
    assert amplitude / 2 == pytest.approx(0.16, abs=5e-3)
    assert retard_de_phase_en_degres(1.0, 20.0) == pytest.approx(9.0)
    assert pire_ecart_sur_une_periode(1.0, 20.0) == pytest.approx(amplitude, rel=1e-4)


def test_l_exemple_2_7_deux_poles_doubles() -> None:
    """Page 53 : `(s + 3) / ((s + 1)^2 (s + 2)^2)` ; -3, 2, 3, 1."""
    transfert = (S + 3) / ((S + 1) ** 2 * (S + 2) ** 2)
    attendue = -3 / (S + 1) + 2 / (S + 1) ** 2 + 3 / (S + 2) + 1 / (S + 2) ** 2
    assert sp.simplify(transfert - attendue) == 0
    fractions = decomposition_en_fractions_simples(transfert, S)
    assert set(fractions) == set(attendue.as_ordered_terms())
    assert ((S + 1) ** 2 * transfert).subs(S, -1) == 2
    assert ((S + 2) ** 2 * transfert).subs(S, -2) == 1
    systeme = sp.Matrix([[1, sp.Rational(1, 2)], [-sp.Rational(1, 2), -1]])
    membre = sp.Matrix([-sp.Rational(3, 2), -sp.Rational(3, 2)])
    assert systeme.solve(membre) == sp.Matrix([-3, 3])
    reponse = reponse_impulsionnelle(transfert, S, T)
    imprimee = (-3 + 2 * T) * sp.exp(-T) + (3 + T) * sp.exp(-2 * T)
    assert sp.simplify(reponse - imprimee) == 0
    developpe = S**4 + 6 * S**3 + 13 * S**2 + 12 * S + 4
    assert sp.expand((S + 1) ** 2 * (S + 2) ** 2) == developpe


GM, RPI, RE, RC = sp.symbols("g_m r_pi R_E R_C", positive=True)


def test_l_exemple_2_10_le_thevenin_de_l_emetteur_commun() -> None:
    """Page 61 : `V_th = -R_C g_m r_pi V_in / (R_E (1 + g_m r_pi) + r_pi)`."""
    elements = (
        SourceDeTension(1, 0, sp.Integer(1), sp.Symbol("i_e")),
        Admittance(1, 2, 1 / RPI),
        Admittance(2, 0, 1 / RE),
        Admittance(3, 0, 1 / RC),
        SourceCommandee(3, 2, GM, 1, 2),
    )
    tensions = resoudre(elements, 3)
    attendue = -RC * GM * RPI / (RE * (1 + GM * RPI) + RPI)
    assert sp.simplify(tensions[2] - attendue) == 0
    emetteur = (1 / RPI + GM) / (1 / RPI + GM + 1 / RE)
    assert sp.simplify(tensions[1] - emetteur) == 0
    sans_degenerescence = sp.limit(tensions[2], RE, 0)
    assert sp.simplify(sans_degenerescence + GM * RC) == 0


def test_l_impedance_de_thevenin_de_l_emetteur_commun_est_r_c() -> None:
    """Page 62 : entree a la masse, `V_E = 0`, la source commandee s'eteint."""
    elements = (
        SourceDeCourant(0, 2, sp.Integer(1)),
        Admittance(1, 0, 1 / RPI),
        Admittance(1, 0, 1 / RE),
        Admittance(2, 0, 1 / RC),
        SourceCommandee(2, 1, GM, 0, 1),
    )
    tensions = resoudre(elements, 2)
    assert sp.simplify(tensions[0]) == 0
    assert sp.simplify(tensions[1] - RC) == 0


def test_l_exemple_2_13_le_circuit_rlc_charge() -> None:
    """Page 67 : `v_0 (1 - e^(-a t / T_0) sin(b t / T_0 + phi) / b)`, `T_0^2 = L C`."""
    v0, amorti, periode = sp.symbols("v_0 alpha T_0", positive=True)
    racine = sp.sqrt(1 - amorti**2)
    dephasage = sp.atan(sp.sqrt(1 / amorti**2 - 1))
    oscillation = sp.sin(racine * T / periode + dephasage)
    imprimee = v0 - v0 / racine * sp.exp(-amorti * T / periode) * oscillation
    valeurs = {v0: 3, amorti: sp.Rational(3, 5), periode: 2}
    reponse = imprimee.subs(valeurs)
    amortissement = 2 * sp.Rational(3, 5) * 2 * sp.diff(reponse, T)
    equation = 4 * sp.diff(reponse, T, 2) + amortissement
    assert sp.simplify(equation + reponse - 3) == 0
    assert sp.simplify(reponse.subs(T, 0)) == 0
    assert sp.simplify(sp.diff(reponse, T).subs(T, 0)) == 0
    assert sp.limit(reponse, T, sp.oo) == 3


def test_les_poles_de_l_exemple_2_13() -> None:
    """Page 66 : `(-a +- racine(a^2 - 1)) / racine(L C)`, `a = R racine(C / L) / 2`."""
    resistance, inductance, capacite = sp.symbols("R L C", positive=True)
    amorti = resistance / 2 * sp.sqrt(capacite / inductance)
    propre = 1 / sp.sqrt(inductance * capacite)
    denominateur = S**2 + S * resistance / inductance + propre**2
    for signe in (-1, 1):
        pole = (-amorti + signe * sp.sqrt(amorti**2 - 1)) * propre
        assert sp.simplify(denominateur.subs(S, pole)) == 0
    discriminant = sp.sqrt(resistance**2 - 4 * inductance / capacite)
    brut = (-resistance + discriminant) / (2 * inductance)
    assert sp.simplify(denominateur.subs(S, brut)) == 0
