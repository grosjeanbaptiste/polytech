"""Le syllabus de signaux et systemes, chapitre 5 : les exercices resolus.

Source : `SignauxSystemes/SYLLABUS/Syllabus_2019_Chapitre_05_Analyse_temporelle.pdf`,
pages 25 a 31. Les oracles sont les solutions imprimees ou manuscrites :
« periode 2 pi s, amorti apres 5 s » (5.1), le courant de l'exercice 5.2,
`R > racine(L / C) / 2` (5.3), `H = 10^4 / (p + 10^4)` (5.4),
`alpha >= 20 racine(2)` (5.5), les limites 0 et 1/2 (5.6) et les raies de
sortie de l'exercice 5.7.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from polytech.electronique.domain.laplace import (
    diviseur_de_tension,
    impedance_de_bobine,
    impedance_de_condensateur,
    impedances_en_parallele,
)
from polytech.maths.domain.laplace import theoremes
from polytech.maths.domain.symboles import IMAGE as P_DES_MATHS
from polytech.signaux.domain.fluence import transmittance_du_graphe
from polytech.signaux.domain.heaviside import original
from polytech.signaux.domain.premier_ordre import PremierOrdre
from polytech.signaux.domain.raies import raie_de_sortie
from polytech.signaux.domain.reponses_du_second_ordre import periode_propre
from polytech.signaux.domain.second_ordre import SecondOrdre
from polytech.signaux.domain.symboles import IMAGE, TEMPS
from polytech.signaux.domain.transformees import transformee_de_laplace

P, T = IMAGE, TEMPS


def test_l_exercice_5_1_la_tension_d_une_capacite_sans_inverser() -> None:
    """`p / (p^2 + 2 p + 2)` : `sigma = 1`, `rho = racine(2)`, donc `omega = 1`.

    « Oscillatoire de periode 2 pi s, amorti apres 5 s : meme pas une
    periode. »
    """
    systeme = SecondOrdre.depuis_evans(1, 1, sp.sqrt(2))
    assert systeme.regime() == "poles complexes conjugues"
    assert periode_propre(systeme) == 2 * sp.pi
    _, sigma, _ = systeme.d_evans()
    assert 5 / sigma < periode_propre(systeme)
    image = (P / (P**2 + 2 * P + 2)).subs(P, P_DES_MATHS)
    assert theoremes.valeur_initiale(image) == 1
    assert theoremes.valeur_finale(image) == 0


def test_l_exercice_5_2_le_courant_d_un_circuit_rl_sous_tension_sinusoidale() -> None:
    """`i = Y E racine(2) (cos(w t + phi + theta) - cos(phi + theta) exp(-t / tau))`."""
    resistance, inductance, phase = 10, sp.Rational(1, 10), sp.pi / 6
    pulsation, crete = 100 * sp.pi, 220 * sp.sqrt(2)
    circuit = PremierOrdre(
        gain=sp.Rational(1, resistance), constante=inductance / resistance
    )
    courant = crete * circuit.reponse_au_cosinus(pulsation, phase=phase)
    source = crete * sp.cos(pulsation * T + phase)
    equation = inductance * sp.diff(courant, T) + resistance * courant - source
    assert sp.simplify(equation) == 0
    assert sp.simplify(courant.subs(T, 0)) == 0


def test_l_exercice_5_2_la_crete_peut_doubler_si_phi_plus_theta_vaut_k_pi() -> None:
    """Le transitoire s'ajoute au regime : jusqu'a deux fois sa crete."""
    circuit = PremierOrdre(gain=1, constante=sp.Rational(1, 2))
    pulsation = 200 * sp.pi
    gain, dephasage = circuit.regime_harmonique(pulsation)
    defavorable = circuit.reponse_au_cosinus(pulsation, phase=-dephasage + sp.pi)
    instants = np.linspace(0, 0.02, 4001)
    valeurs = sp.lambdify(T, defavorable, "numpy")(instants)
    assert np.max(np.abs(valeurs)) == pytest.approx(2 * float(gain), rel=0.02)
    favorable = circuit.reponse_au_cosinus(pulsation, phase=-dephasage + sp.pi / 2)
    valeurs = sp.lambdify(T, favorable, "numpy")(instants)
    assert np.max(np.abs(valeurs)) == pytest.approx(float(gain), rel=0.01)


def test_l_exercice_5_3_le_circuit_rlc_parallele_oscille_si_r_est_assez_grand() -> None:
    """`R > racine(L / C) / 2` ; alors `tau = 2 R C`."""
    resistance, inductance, capacite = sp.symbols("R L C", positive=True)
    impedance = impedances_en_parallele(
        resistance,
        impedance_de_bobine(inductance, P),
        impedance_de_condensateur(capacite, P),
    )
    denominateur = sp.Poly(sp.fraction(sp.cancel(impedance))[1], P).monic()
    sigma, rho_carre = (
        denominateur.coeff_monomial(P) / 2,
        denominateur.coeff_monomial(1),
    )
    assert sp.simplify(sigma - 1 / (2 * resistance * capacite)) == 0
    assert sp.simplify(rho_carre - 1 / (inductance * capacite)) == 0
    limite = sp.sqrt(inductance / capacite) / 2
    assert sp.simplify((sigma**2 - rho_carre).subs(resistance, limite)) == 0
    assert sp.simplify((sigma**2 - rho_carre).subs(resistance, 2 * limite)).is_negative
    assert sp.simplify(1 / sigma - 2 * resistance * capacite) == 0


def test_l_exercice_5_4_la_capacite_en_tete_ne_compte_pas() -> None:
    """`H = R / (p L + R) = 10^4 / (p + 10^4)`, `h = 10^4 exp(-t / tau)`, 0,1 ms."""
    resistance, inductance = 10, sp.Rational(1, 1000)
    transmittance = diviseur_de_tension(impedance_de_bobine(inductance, P), resistance)
    assert sp.simplify(transmittance - 10**4 / (P + 10**4)) == 0
    circuit = PremierOrdre(gain=1, constante=inductance / resistance)
    assert circuit.d_evans() == (10**4, 10**4)
    assert sp.simplify(original(transmittance) - circuit.impulsionnelle()) == 0
    assert circuit.constante == sp.Rational(1, 10**4)


def test_l_exercice_5_4_les_impulsions_du_train_ne_se_recouvrent_pas() -> None:
    """Periode 10 ms, cent fois la constante de temps : chaque reponse s'eteint."""
    circuit = PremierOrdre(gain=1, constante=sp.Rational(1, 10**4))
    periode = sp.Rational(1, 100)
    assert periode / circuit.constante == 100
    reste = circuit.impulsionnelle().subs(T, periode) / circuit.impulsionnelle().subs(
        T, 0
    )
    assert float(reste) < 1e-40


def test_l_exercice_5_5_la_valeur_d_alpha_qui_supprime_l_oscillation() -> None:
    """Mason : `p / (p^2 + alpha p + 200)` ; aperiodique si `alpha >= 20 racine(2)`."""
    alpha = sp.Symbol("alpha", positive=True)
    transmittance = transmittance_du_graphe(retours=(-alpha, -200), directs=(1, 0))
    assert sp.simplify(transmittance - P / (P**2 + alpha * P + 200)) == 0
    limite = 20 * sp.sqrt(2)
    critique = SecondOrdre.depuis_evans(1, limite / 2, sp.sqrt(200))
    assert critique.regime() == "poles reels confondus"
    assert (
        SecondOrdre.depuis_evans(1, 10, sp.sqrt(200)).regime()
        == "poles complexes conjugues"
    )
    assert (
        SecondOrdre.depuis_evans(1, 20, sp.sqrt(200)).regime()
        == "poles reels distincts"
    )


def test_l_exercice_5_6_la_reponse_indicielle_du_circuit_de_l_exemple_4_6() -> None:
    """1 ohm, 1 mF, 1 mH : `a` part de 0 et tend vers 1/2."""
    millieme = sp.Rational(1, 1000)
    transmittance = 1 / (P**2 * millieme**2 + P * 2 * millieme + 2)
    indicielle = (transmittance / P).subs(P, P_DES_MATHS)
    assert theoremes.valeur_initiale(indicielle) == 0
    assert theoremes.valeur_finale(indicielle) == sp.Rational(1, 2)
    reponse = original(transmittance / P)
    assert reponse.subs(T, 0) == 0
    assert sp.limit(reponse, T, sp.oo) == sp.Rational(1, 2)


def test_l_exercice_5_7_chaque_raie_est_multipliee_par_la_reponse_en_frequence() -> (
    None
):
    """`Y_k = (1 / (1 + j k)) (1 / (1 + j k omega_0))`, pour `h = exp(-t)`."""
    transmittance = transformee_de_laplace(sp.exp(-T))
    assert transmittance == 1 / (P + 1)
    fondamentale = 2 * sp.pi * 10
    for rang in (-2, 0, 1, 3):
        entree = 1 / (1 + sp.I * rang)
        sortie = raie_de_sortie(entree, transmittance, rang, fondamentale)
        attendue = entree / (1 + sp.I * rang * fondamentale)
        assert sp.simplify(sortie - attendue) == 0
    with pytest.raises(ValueError, match="pole sur la raie"):
        raie_de_sortie(1, 1 / (P**2 + 4), 1, 2)
