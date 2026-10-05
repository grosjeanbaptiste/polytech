"""Les planches d'electronique : Laplace, equivalents, phaseurs, Bode (1, III).

Source : `ElectroniqueInstrumentation/SLIDES/Slides_1_part3.pdf`, 71
planches lues sur les pages rendues -- leurs formules sont des images.

La transformee de Laplace et celle de Fourier sont des mathematiques, et
le trace asymptotique est celui du cours de signaux : ce fichier relie
les trois cours et eprouve les exemples chiffres des planches -- les
decompositions des planches 23, 26 et 28, l'impedance de la planche 19,
la transmittance des planches 62 et 71.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electronique.domain.bode_asymptotique import (
    erreur_de_l_asymptote,
    erreur_de_phase_de_l_asymptote,
    phase_asymptotique,
    pic_du_second_ordre,
)
from polytech.electronique.domain.conditions_initiales import (
    courant_de_la_bobine,
    tension_de_la_bobine,
    tension_du_condensateur,
)
from polytech.electronique.domain.equivalents import (
    EquivalentDeNorton,
    EquivalentDeThevenin,
    diviseur_generalise,
    thevenin_du_diviseur,
)
from polytech.electronique.domain.laplace import (
    decomposition_en_fractions_simples,
    diviseur_de_courant,
    diviseur_de_tension,
    impedance_de_bobine,
    impedance_de_condensateur,
    impedance_de_resistance,
    impedances_en_parallele,
    impedances_en_serie,
    multiplicites_des_poles,
    reponse_impulsionnelle,
    residu_en_un_pole,
)
from polytech.electronique.domain.phaseurs import (
    phaseur,
    reponse_asymptotique,
    signal_du_phaseur,
    somme_de_sinus,
)
from polytech.maths.domain.fourier.proprietes import (
    energie_frequentielle,
    energie_temporelle,
)
from polytech.maths.domain.fourier.transformee import (
    PULSATION,
    transformee_de_fourier,
)
from polytech.maths.domain.laplace.proprietes import (
    de_la_derivee_n,
    du_produit_par_une_exponentielle,
)
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import IMAGE, TEMPS, VARIABLE
from polytech.signaux.domain.bode import TraceAsymptotique

S = sp.Symbol("s")
T = sp.Symbol("t", positive=True)
W = sp.Symbol("omega", positive=True)
R, C, L = sp.symbols("R C L", positive=True)


def test_la_table_des_transformees_de_laplace() -> None:
    """Planche 8 : `1/s`, `1/s^2`, `n!/s^(n+1)`, `w/(s^2+w^2)`, `s/(s^2+w^2)`."""
    attendues = {
        sp.Integer(1): 1 / IMAGE,
        TEMPS: 1 / IMAGE**2,
        TEMPS**3: 6 / IMAGE**4,
        sp.sin(W * TEMPS): W / (IMAGE**2 + W**2),
        sp.cos(W * TEMPS): IMAGE / (IMAGE**2 + W**2),
    }
    for original, image in attendues.items():
        assert sp.simplify(Transformee.de(original).image - image) == 0


def test_les_deux_ecritures_du_sinus_et_du_cosinus() -> None:
    """Planche 8 : `s/(s^2+w^2) = (1/2)[1/(s-jw) + 1/(s+jw)]`, et le sinus en `j`."""
    j = sp.I
    cosinus = (1 / (S - j * W) + 1 / (S + j * W)) / 2
    sinus = -(j / (S - j * W) - j / (S + j * W)) / 2
    assert sp.simplify(cosinus - S / (S**2 + W**2)) == 0
    assert sp.simplify(sinus - W / (S**2 + W**2)) == 0


def test_les_proprietes_de_la_transformee() -> None:
    """Planche 7 : decalage frequentiel, derivees premiere et seconde."""
    image = Transformee.de(sp.cos(W * TEMPS)).image
    amortie = Transformee.de(sp.exp(-2 * TEMPS) * sp.cos(W * TEMPS)).image
    assert sp.simplify(du_produit_par_une_exponentielle(image, -2) - amortie) == 0
    depart, pente = sp.symbols("f_0 f_1")
    generique = sp.Function("F")(IMAGE)
    assert de_la_derivee_n(generique, [depart]) == IMAGE * generique - depart
    seconde = de_la_derivee_n(generique, [depart, pente])
    assert sp.expand(seconde - (IMAGE**2 * generique - IMAGE * depart - pente)) == 0


def test_le_cosinus_amorti_et_dephase() -> None:
    """Planche 9 : `A e^(at) cos(wt + phi)` en deux fractions conjuguees."""
    amplitude, amortissement, phase = sp.Rational(3), sp.Rational(-1, 2), sp.pi / 3
    original = amplitude * sp.exp(amortissement * TEMPS) * sp.cos(2 * TEMPS + phase)
    j = sp.I
    planche = (
        amplitude * sp.exp(j * phase) / (IMAGE - (amortissement + 2 * j))
        + amplitude * sp.exp(-j * phase) / (IMAGE - (amortissement - 2 * j))
    ) / 2
    image = Transformee.de(original).image
    for valeur in (0.5, 2.0, 7.0):
        attendue = complex(planche.subs(IMAGE, valeur))
        assert complex(image.subs(IMAGE, valeur)) == pytest.approx(attendue)


def test_les_impedances_et_leurs_conditions_initiales() -> None:
    """Planches 11 et 12 : `V = I/(sC) + v(0-)/s` et `V = sL I - L i(0-)`."""
    courant, initiale = sp.symbols("I v_0")
    condensateur = tension_du_condensateur(C, courant, S, initiale=initiale)
    assert sp.simplify(S * C * condensateur - C * initiale - courant) == 0
    assert tension_du_condensateur(
        C, courant, S
    ) == courant * impedance_de_condensateur(C, S)
    bobine = tension_de_la_bobine(L, courant, S, initial=initiale)
    assert sp.expand(bobine - (S * L * courant - L * initiale)) == 0
    assert tension_de_la_bobine(L, courant, S) == courant * impedance_de_bobine(L, S)
    tension = sp.Symbol("V")
    retour = courant_de_la_bobine(L, tension, S, initial=initiale)
    assert (
        sp.simplify(tension_de_la_bobine(L, retour, S, initial=initiale) - tension) == 0
    )


def test_les_associations_d_impedances_et_d_admittances() -> None:
    """Planche 13 : serie `Z_1 + Z_2`, parallele `1 / (1/Z_1 + 1/Z_2)`, `Y = 1/Z`."""
    un, deux = sp.symbols("Z_1 Z_2", positive=True)
    assert impedances_en_serie(un, deux) == un + deux
    parallele = impedances_en_parallele(un, deux)
    assert sp.simplify(parallele - 1 / (1 / un + 1 / deux)) == 0
    assert sp.simplify(1 / parallele - (1 / un + 1 / deux)) == 0


def test_l_impedance_de_la_planche_19() -> None:
    """`R` et `C` en serie, en parallele sur `L` : `sL(sRC+1)/(s^2 LC+sRC+1)`."""
    branche = impedances_en_serie(
        impedance_de_resistance(R, S), impedance_de_condensateur(C, S)
    )
    assert sp.simplify(branche - (S * R * C + 1) / (S * C)) == 0
    totale = impedances_en_parallele(branche, impedance_de_bobine(L, S))
    planche = (S**2 * L * R * C + S * L) / (S**2 * L * C + S * R * C + 1)
    assert sp.simplify(totale - planche) == 0


def test_la_decomposition_a_poles_simples_de_la_planche_23() -> None:
    """`(s + 1) / ((s + 2)(s + 3))` : `c_1 = -1`, `c_2 = 2`."""
    transfert = (S + 1) / ((S + 2) * (S + 3))
    assert residu_en_un_pole(transfert, S, sp.Integer(-2)) == -1
    assert residu_en_un_pole(transfert, S, sp.Integer(-3)) == 2
    fractions = decomposition_en_fractions_simples(transfert, S)
    assert sp.simplify(sum(fractions) - transfert) == 0
    reponse = reponse_impulsionnelle(transfert, S, T)
    assert sp.simplify(reponse - (-sp.exp(-2 * T) + 2 * sp.exp(-3 * T))) == 0


def test_la_decomposition_a_poles_conjugues_de_la_planche_26() -> None:
    """`1 / ((s^2 - 2s + 2)(s + 1))` : `c_1 = j (j - 2) / 10`, `c_2 = 0,2`.

    `K = 1 / (2 racine(5))`, `phi = -pi/2 - arctan(0,5)`, et
    `h(t) = 2 K e^t cos(t + phi) + 0,2 e^(-t)`.
    """
    transfert = 1 / ((S**2 - 2 * S + 2) * (S + 1))
    conjugue = residu_en_un_pole(transfert, S, 1 + sp.I)
    assert sp.simplify(conjugue - sp.I * (sp.I - 2) / 10) == 0
    module, argument = sp.Abs(conjugue), sp.arg(conjugue)
    assert sp.simplify(module - 1 / (2 * sp.sqrt(5))) == 0
    assert float(argument) == pytest.approx(-math.pi / 2 - math.atan(0.5))
    assert residu_en_un_pole(transfert, S, sp.Integer(-1)) == sp.Rational(1, 5)
    planche = 2 * module * sp.exp(T) * sp.cos(T + argument) + sp.exp(-T) / 5
    reponse = reponse_impulsionnelle(transfert, S, T)
    for instant in (0.0, 0.4, 1.3):
        assert float(reponse.subs(T, instant)) == pytest.approx(
            float(planche.subs(T, instant))
        )


def test_la_decomposition_a_poles_doubles_de_la_planche_28() -> None:
    """`(s + 3) / ((s + 1)^2 (s + 2)^2)` : `-3`, `2`, `3`, `1`."""
    transfert = (S + 3) / ((S + 1) ** 2 * (S + 2) ** 2)
    assert multiplicites_des_poles(transfert, S) == {-1: 2, -2: 2}
    planche = -3 / (S + 1) + 2 / (S + 1) ** 2 + 3 / (S + 2) + 1 / (S + 2) ** 2
    assert sp.simplify(planche - transfert) == 0
    assert (transfert * (S + 1) ** 2).subs(S, -1) == 2
    assert (transfert * (S + 2) ** 2).subs(S, -2) == 1
    systeme = sp.Matrix([[1, sp.Rational(1, 2)], [-sp.Rational(1, 2), -1]])
    membre = sp.Matrix([-sp.Rational(3, 2), -sp.Rational(3, 2)])
    assert systeme.solve(membre) == sp.Matrix([-3, 3])
    reponse = reponse_impulsionnelle(transfert, S, T)
    attendue = (
        -3 * sp.exp(-T) + 2 * T * sp.exp(-T) + 3 * sp.exp(-2 * T) + T * sp.exp(-2 * T)
    )
    assert sp.simplify(reponse - attendue) == 0


def test_le_diviseur_de_tension_generalise() -> None:
    """Planches 33 et 34 : `V_out = Z_2 / (Z_1 + Z_2) (V_top - V_bottom) + V_bottom`."""
    un, deux, haut, bas = sp.symbols("Z_1 Z_2 V_top V_bottom")
    sortie = diviseur_generalise(un, deux, haut, bas)
    assert sp.simplify(sortie - (deux / (un + deux) * (haut - bas) + bas)) == 0
    assert sp.simplify(haut - sortie - un / (un + deux) * (haut - bas)) == 0
    classique = diviseur_generalise(un, deux, haut)
    assert sp.simplify(classique - haut * diviseur_de_tension(un, deux)) == 0
    assert sp.limit(classique.subs({un: 1, haut: 5}), deux, sp.oo) == 5


def test_le_diviseur_de_courant() -> None:
    """Planche 35 : `I = Y / (Y_par + Y) I_tot` ; le courant va au plus facile."""
    branche, autre = sp.symbols("Z Z_par", positive=True)
    part = diviseur_de_courant(branche, autre)
    assert sp.simplify(part - (1 / branche) / (1 / autre + 1 / branche)) == 0
    assert part.subs({branche: 1, autre: 9}) == sp.Rational(9, 10)


def test_thevenin_et_norton_se_correspondent() -> None:
    """Planche 39 : `Z_th = Z_no`, `V_th = I_no Z_no`, `I_no = V_th / Z_th`."""
    tension, impedance = sp.symbols("V_th Z_th", positive=True)
    thevenin = EquivalentDeThevenin(tension, impedance)
    norton = thevenin.norton()
    assert norton == EquivalentDeNorton(tension / impedance, impedance)
    assert norton.thevenin() == thevenin
    charge = sp.Symbol("Z_L", positive=True)
    par_thevenin = thevenin.tension_sur(charge)
    par_norton = norton.courant * impedances_en_parallele(impedance, charge)
    assert sp.simplify(par_thevenin - par_norton) == 0
    with pytest.raises(ValueError, match="impedance de Thevenin nulle"):
        EquivalentDeThevenin(tension, sp.Integer(0)).norton()


def test_le_thevenin_du_circuit_rc_de_la_planche_38() -> None:
    """`V_th = V_in / (1 + sRC)`, `Z_th = R / (1 + sRC)`."""
    entree = sp.Symbol("V_in")
    equivalent = thevenin_du_diviseur(entree, R, impedance_de_condensateur(C, S))
    assert sp.simplify(equivalent.tension - entree / (1 + S * R * C)) == 0
    assert sp.simplify(equivalent.impedance - R / (1 + S * R * C)) == 0


def test_le_thevenin_ne_predit_pas_la_puissance_interne() -> None:
    """Planches 36 et 37 : meme loi courant-tension pour la charge, pas en dedans.

    Une source `V_in` en parallele sur `R` : `V_th = V_in`, `Z_th = 0`.
    Le circuit d'origine dissipe `V_in^2 / R` dans sa resistance, charge
    ou non ; l'equivalent n'en dissipe aucune.
    """
    equivalent = EquivalentDeThevenin(sp.Integer(12), sp.Integer(0))
    charge = sp.Integer(6)
    assert equivalent.tension_sur(charge) == 12
    assert equivalent.puissance_interne(charge) == 0
    origine = sp.Integer(12) ** 2 / sp.Integer(4)
    assert origine == 36
    reel = EquivalentDeThevenin(sp.Integer(12), sp.Integer(2))
    assert reel.puissance_interne(charge) == sp.Rational(9, 2)


def test_le_filtre_rc_sous_un_cosinus_par_laplace_et_par_les_phaseurs() -> None:
    """Planches 41 et 55 : la reponse complete, et sa partie qui ne s'eteint pas."""
    tau, pulsation, amplitude = sp.Rational(1, 2), sp.Integer(3), sp.Integer(5)
    transfert = 1 / (1 + S * tau)
    entree = amplitude * S / (S**2 + pulsation**2)
    complete = reponse_impulsionnelle(transfert * entree, S, T)
    module, phase = reponse_asymptotique(transfert, S, pulsation, amplitude=amplitude)
    assert sp.simplify(module - amplitude / sp.sqrt(1 + (pulsation * tau) ** 2)) == 0
    assert sp.simplify(phase + sp.atan(pulsation * tau)) == 0
    permanente = module * sp.cos(pulsation * T + phase)
    transitoire = sp.simplify(complete - permanente)
    attendue = -amplitude / (1 + (pulsation * tau) ** 2) * sp.exp(-T / tau)
    for instant in (0.0, 0.3, 1.1):
        assert float(transitoire.subs(T, instant)) == pytest.approx(
            float(attendue.subs(T, instant))
        )
    assert float(complete.subs(T, 0)) == pytest.approx(0.0, abs=1e-12)


def test_le_coefficient_du_pole_de_l_entree() -> None:
    """Planche 58 : `b = Y(s)(s - j w_0)` en `s = j w_0` vaut `H(j w_0) A / 2`."""
    transfert = (S + 2) / (S**2 + 3 * S + 5)
    amplitude, pulsation = sp.Integer(4), sp.Integer(2)
    sortie = transfert * amplitude * S / (S**2 + pulsation**2)
    coefficient = residu_en_un_pole(sortie, S, sp.I * pulsation)
    attendu = transfert.subs(S, sp.I * pulsation) * amplitude / 2
    assert sp.simplify(coefficient - attendu) == 0


def test_laplace_et_fourier_coincident_pour_un_signal_causal_et_stable() -> None:
    """Planches 44 et 45 : `s = j 2 pi f` ; `e^(-at) u(t)` donne `1 / (j w + a)`."""
    signal = sp.exp(-3 * VARIABLE) * sp.Heaviside(VARIABLE)
    fourier = transformee_de_fourier(signal)
    laplace = Transformee.de(sp.exp(-3 * TEMPS)).image
    assert sp.simplify(fourier - laplace.subs(IMAGE, sp.I * PULSATION)) == 0
    assert sp.simplify(fourier - 1 / (sp.I * PULSATION + 3)) == 0


def test_le_theoreme_de_rayleigh() -> None:
    """Planche 47 : `integrale |g|^2 dt = integrale |G|^2 df`."""
    signal = sp.exp(-3 * VARIABLE) * sp.Heaviside(VARIABLE)
    temporelle = energie_temporelle(signal)
    assert temporelle == sp.Rational(1, 6)
    assert energie_frequentielle(transformee_de_fourier(signal)) == temporelle


def test_un_phaseur_represente_un_cosinus() -> None:
    """Planche 49 : `A cos(wt + phi) = Re[A e^(j phi) e^(j w t)]`."""
    complexe = phaseur(sp.Integer(2), sp.pi / 6)
    signal = signal_du_phaseur(complexe, W, T)
    assert sp.simplify(signal - 2 * sp.cos(W * T + sp.pi / 6)) == 0
    derive = signal_du_phaseur(sp.I * W * complexe, W, T)
    assert sp.simplify(derive - sp.diff(signal, T)) == 0
    with pytest.raises(ValueError, match="amplitude negative"):
        phaseur(sp.Integer(-2), sp.pi / 6)


def test_une_somme_de_sinus_de_meme_pulsation_est_un_sinus() -> None:
    """Planche 49 : `c^2 = a^2 + b^2 + 2ab cos(theta_a - theta_b)`, et `tan(phi)`."""
    a, theta_a, b, theta_b = 3.0, 0.4, 2.0, -1.1
    c, phi = somme_de_sinus(a, theta_a, b, theta_b)
    assert c**2 == pytest.approx(a**2 + b**2 + 2 * a * b * math.cos(theta_a - theta_b))
    numerateur = a * math.sin(theta_a) + b * math.sin(theta_b)
    denominateur = a * math.cos(theta_a) + b * math.cos(theta_b)
    assert math.tan(phi) == pytest.approx(numerateur / denominateur)
    for instant in (0.0, 0.7, 2.9):
        somme = a * math.sin(instant + theta_a) + b * math.sin(instant + theta_b)
        assert c * math.sin(instant + phi) == pytest.approx(somme)


def test_les_impedances_harmoniques() -> None:
    """Planches 51 et 52 : `V = j w L I` et `V = I / (j w C)`."""
    assert impedance_de_bobine(L, sp.I * W) == sp.I * W * L
    assert sp.simplify(impedance_de_condensateur(C, sp.I * W) - 1 / (sp.I * W * C)) == 0


EXEMPLE = 10**4 * (S + 10**3) / ((S + 10**5) * (S + 10))
TRACE = TraceAsymptotique(
    gain=10.0, classe=0, cassures=((10.0, -1), (1e3, 1), (1e5, -1))
)


def _decibels(pulsation: float) -> float:
    return 20 * math.log10(abs(complex(EXEMPLE.subs(S, 1j * pulsation))))


def test_la_transmittance_des_planches_62_et_71() -> None:
    """Denominateur `s^2 + 100010 s + 1e6` ; 20 dB en continu, puis -20 dB."""
    denominateur = sp.expand((S + 10**5) * (S + 10))
    assert denominateur == S**2 + 100010 * S + 10**6
    assert sp.expand(10**4 * (S + 10**3)) == 10**4 * S + 10**7
    assert _decibels(1e-2) == pytest.approx(20.0, abs=0.01)
    assert TRACE.pentes() == (0, -20, 0, -20)
    assert TRACE.gain_en_decibels(1.0) == pytest.approx(20.0)
    assert TRACE.gain_en_decibels(1e4) == pytest.approx(-20.0)
    assert TRACE.gain_en_decibels(1e6) == pytest.approx(-40.0)
    for pulsation in (1.0, 1e2, 1e4, 1e6):
        assert _decibels(pulsation) == pytest.approx(
            TRACE.gain_en_decibels(pulsation), abs=0.1
        )


def test_la_precision_des_asymptotes() -> None:
    """Planches 66 et 67 : justes sous `b/4` et au-dela de `4b` ; `b/10` et `10b`.

    A la cassure, l'ecart est de 3 dB ; a un quart ou a quatre fois la
    cassure, de 0,26 dB. Pour la phase, a un dixieme ou a dix fois, 5,7
    degres.
    """
    assert erreur_de_l_asymptote(1.0) == pytest.approx(3.01, abs=0.005)
    assert erreur_de_l_asymptote(0.25) == pytest.approx(0.26, abs=0.005)
    assert erreur_de_l_asymptote(4.0) == pytest.approx(0.26, abs=0.005)
    assert erreur_de_phase_de_l_asymptote(0.1) == pytest.approx(5.7, abs=0.02)
    assert erreur_de_phase_de_l_asymptote(10.0) == pytest.approx(5.7, abs=0.02)
    assert erreur_de_phase_de_l_asymptote(1.0) == pytest.approx(45.0)
    with pytest.raises(ValueError, match="rapport nul ou negatif"):
        erreur_de_l_asymptote(0.0)


def test_la_phase_asymptotique_de_l_exemple() -> None:
    """Planche 71 : 0, puis -90, 0 et -90 degres, en marches aux cassures."""
    zeros, poles = (1e3,), (10.0, 1e5)
    marches = [
        phase_asymptotique(w, zeros=zeros, poles=poles) for w in (1, 100, 1e4, 1e6)
    ]
    assert marches == [0.0, -90.0, 0.0, -90.0]
    exacte = math.degrees(sp.arg(EXEMPLE.subs(S, 100j)))
    assert exacte == pytest.approx(-78.6, abs=0.1)


def test_le_second_ordre_et_son_pic() -> None:
    """Planches 68 et 70 : en `w_0`, `-j / (2 xi w_0^2)` ; pic de `20 log(1 / 2 xi)`."""
    amortissement, propre = sp.Rational(1, 10), sp.Integer(1000)
    paire = 1 / ((sp.I * W) ** 2 + 2 * sp.I * amortissement * propre * W + propre**2)
    au_pic = paire.subs(W, propre)
    assert sp.simplify(au_pic + sp.I / (2 * amortissement * propre**2)) == 0
    normalise = abs(complex(au_pic * propre**2))
    assert 20 * math.log10(normalise) == pytest.approx(pic_du_second_ordre(0.1))
    assert pic_du_second_ordre(0.1) == pytest.approx(13.98, abs=0.005)
    assert pic_du_second_ordre(0.5) == pytest.approx(0.0)
    poles = sp.solve(S**2 + 2 * amortissement * propre * S + propre**2, S)
    attendu = propre * (-amortissement + sp.I * sp.sqrt(1 - amortissement**2))
    assert any(sp.simplify(pole - attendu) == 0 for pole in poles)
    with pytest.raises(ValueError, match="amortissement nul ou negatif"):
        pic_du_second_ordre(0.0)
