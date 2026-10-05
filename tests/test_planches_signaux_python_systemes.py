"""Planches de signaux et systemes : transmittances et analyse temporelle.

Sources : `SignauxSystemes/SLIDES/Slides_Python_04_Transmittances.pdf`
(28 pages) et `Slides_Python_05_Analyse_temporelle.pdf` (29 pages), lues
sur les pages rendues. Elles projettent les chapitres 4 et 5, deja portes
par la bibliotheque : ce fichier les y relie, et etablit quatre ecarts
qu'un zoom a confirmes -- un signe et une casse dans la regle de Mason,
l'additionneur inverse, un `p^2` perdu, et des poles du mauvais cote.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from polytech.electronique.domain.montages import somme_ponderee
from polytech.signaux.domain.composantes import nature_de_la_composante
from polytech.signaux.domain.fluence import regle_de_mason
from polytech.signaux.domain.forme_generale import mode_d_une_paire_conjuguee
from polytech.signaux.domain.harmonique import AnalyseHarmonique
from polytech.signaux.domain.heaviside import original, residu
from polytech.signaux.domain.modeles import resistance_critique
from polytech.signaux.domain.modes import residus
from polytech.signaux.domain.poles_numeriques import reponse_impulsionnelle
from polytech.signaux.domain.premier_ordre import PremierOrdre
from polytech.signaux.domain.reponses_du_second_ordre import (
    impulsionnelle,
    indicielle,
    periode_propre,
)
from polytech.signaux.domain.second_ordre import SecondOrdre
from polytech.signaux.domain.stabilite_des_poles import nature_de_la_stabilite
from polytech.signaux.domain.suspension import sortie_par_superposition
from polytech.signaux.domain.symboles import IMAGE, PULSATION, TEMPS
from polytech.signaux.domain.temporel import (
    ReponsesTemporelles,
    est_du_troisieme_ordre,
    ordre_du_systeme,
)
from polytech.signaux.domain.transformees import transformee_de_laplace
from polytech.signaux.domain.transmittance import Transmittance

P, T, W = IMAGE, TEMPS, PULSATION
K, TAU = sp.symbols("K T", positive=True)


def test_le_rappel_de_la_planche_4_une_transmittance_par_entree() -> None:
    """`Y = G_u U + G_p P`, autour d'un point de repos."""
    commande, perturbation = sp.symbols("U P_e")
    de_la_commande = Transmittance(expression=2 / (P + 1))
    de_la_perturbation = Transmittance(expression=-1 / (P + 1))
    sortie = sortie_par_superposition(
        {de_la_commande: commande, de_la_perturbation: perturbation}
    )
    assert sp.simplify(sortie - (2 * commande - perturbation) / (P + 1)) == 0


def test_la_transmittance_est_la_transformee_de_la_reponse_impulsionnelle() -> None:
    """Planche 11 : « H(p) est aussi la transformee de Laplace de h(t) »."""
    transmittance = Transmittance(expression=(P + 3) / ((P + 1) * (P + 2)))
    reponse = ReponsesTemporelles(transmittance).impulsionnelle()
    assert sp.simplify(transformee_de_laplace(reponse) - transmittance.expression) == 0


def test_les_formes_generales_de_la_planche_14() -> None:
    """Poles simples : un terme par pole reel, un cosinus amorti par paire."""
    transmittance = (P + 3) / ((P + 1) * (P**2 + 2 * P + 5))
    pole = -1 + 2 * sp.I
    paire = mode_d_une_paire_conjuguee(residu(transmittance, pole), pole)
    reel = residu(transmittance, -1) * sp.exp(-T)
    ecart = sp.lambdify(T, paire + reel - original(transmittance), "numpy")
    assert np.allclose(ecart(np.array([0.0, 0.4, 1.7])), 0, atol=1e-12)


def test_les_deux_ponderations_de_la_planche_17_ont_une_aire_unite() -> None:
    """Diffusion de la chaleur : `exp(-t / tau) / tau`, ou `1 / tau` pendant `tau`."""
    exponentielle = PremierOrdre(gain=1, constante=TAU).impulsionnelle()
    assert sp.integrate(exponentielle, (T, 0, sp.oo)) == 1
    assert sp.integrate(1 / TAU, (T, 0, TAU)) == 1


def test_la_planche_19_signale_elle_meme_le_signe_de_la_regle_de_mason() -> None:
    """« Erreur ! » sur le moins de `-K G / (1 + k G H)` ; le `k` minuscule reste."""
    gain, directe, retour = sp.symbols("K G H")
    fermee = regle_de_mason([gain * directe], [-gain * directe * retour])
    assert sp.simplify(fermee - gain * directe / (1 + gain * directe * retour)) == 0
    minuscule = sp.Symbol("k")
    imprimee = gain * directe / (1 + minuscule * directe * retour)
    assert sp.simplify(fermee - imprimee) != 0
    assert sp.simplify(fermee - imprimee.subs(minuscule, gain)) == 0


def test_l_additionneur_de_la_planche_24_est_imprime_a_l_envers() -> None:
    """« V_out = -(R_1 / R) V_1 - ... » : la figure donne `-(R / R_1)`.

    `R_1` est sur l'entree et `R` en contre-reaction ; la planche
    precedente ecrit elle-meme `H = -Z_2 / Z_1`.
    """
    contre_reaction, voies = 10e3, (2e3, 5e3)
    poids = somme_ponderee(contre_reaction, voies)
    assert poids == pytest.approx((-5.0, -2.0))
    imprimes = tuple(-voie / contre_reaction for voie in voies)
    assert imprimes == pytest.approx((-0.2, -0.5))


def test_les_sorties_de_residue_des_planches_3_a_6() -> None:
    """`r = [-1, 1]`, `r = [0.5 - 0.25j, 0.5 + 0.25j]`, `p = [-2, -1]`."""
    valeurs, poles = residus([1.0], [1.0, 3.0, 2.0])
    assert sorted(poles.real) == pytest.approx([-2.0, -1.0])
    assert sorted(valeurs.real) == pytest.approx([-1.0, 1.0])
    valeurs, poles = residus([1.0, 1.0], [1.0, 0.0, 4.0])
    assert sorted(poles.imag) == pytest.approx([-2.0, 2.0])
    assert sorted(valeurs.imag) == pytest.approx([-0.25, 0.25])
    assert valeurs.real == pytest.approx([0.5, 0.5])


def test_la_decomposition_de_la_reponse_a_la_rampe() -> None:
    """Planche 14 : `K / (p^2 (p T + 1)) = K / p^2 - K T / p + K T / (p + 1 / T)`."""
    systeme = PremierOrdre(gain=K, constante=TAU)
    image = systeme.de_bode().expression / P**2
    decomposee = K / P**2 - K * TAU / P + K * TAU / (P + 1 / TAU)
    assert sp.simplify(image - decomposee) == 0
    reponse = K * T - K * TAU + K * TAU * sp.exp(-T / TAU)
    assert sp.simplify(reponse - systeme.reponse_a_la_rampe()) == 0


def test_le_residu_harmonique_de_la_planche_15() -> None:
    """En `p = j w`, le residu vaut `K / (j w T + 1)`.

    La planche ecrit d'abord « K/T (j w T + 1) », trop petit d'un facteur
    `T`, puis, a la ligne suivante, le bon « (K/T) / (j w + 1/T) ».
    """
    gain, constante, pulsation = 5, 3, 2
    transmittance = PremierOrdre(gain, constante).de_bode().expression
    excitation = pulsation * sp.I
    exact = residu(transmittance / (P - excitation), excitation)
    assert sp.simplify(exact - gain / (excitation * constante + 1)) == 0
    bonne = sp.Rational(gain, constante) / (excitation + sp.Rational(1, constante))
    assert sp.simplify(exact - bonne) == 0
    fautive = sp.Rational(gain, constante) / (excitation * constante + 1)
    assert sp.simplify(exact - constante * fautive) == 0


def test_la_reponse_au_cosinus_de_la_planche_15() -> None:
    """`K / racine(w^2 T^2 + 1) (cos(w t + phi) - exp(-t / T) cos(phi))`."""
    systeme = PremierOrdre(gain=5, constante=3)
    phase = -sp.atan(2 * 3)
    amplitude = 5 / sp.sqrt(2**2 * 3**2 + 1)
    imprimee = amplitude * (sp.cos(2 * T + phase) - sp.exp(-T / 3) * sp.cos(phase))
    assert sp.simplify(imprimee - systeme.reponse_au_cosinus(2)) == 0


def test_la_geometrie_des_poles_de_la_planche_16() -> None:
    """`sin(theta) = xi`, `cos(theta) = racine(1 - xi^2)`, `Q = 1 / (2 xi)`."""
    amortissement = sp.Rational(3, 5)
    systeme = SecondOrdre(5, 2, amortissement)
    angle = systeme.argument_des_poles()
    assert sp.sin(angle) == amortissement
    assert sp.cos(angle) == sp.Rational(4, 5)
    assert sp.tan(angle) == sp.Rational(3, 4)
    assert systeme.facteur_de_qualite() == sp.Rational(5, 6)


def test_le_circuit_rlc_serie_sous_un_echelon_de_la_planche_17() -> None:
    """`I = A C / (p^2 L C + p R C + 1)` ; `T^2 = L C`, `xi = R C / (2 T)`."""
    resistance, inductance, capacite = sp.symbols("R L C", positive=True)
    courant = (1 / P) / (P * inductance + resistance + 1 / (P * capacite))
    denominateur = P**2 * inductance * capacite + P * resistance * capacite + 1
    assert sp.simplify(courant - capacite / denominateur) == 0
    systeme = SecondOrdre(capacite, sp.sqrt(inductance * capacite), sp.S.One)
    amortissement = resistance * capacite / (2 * sp.sqrt(inductance * capacite))
    reconstruit = SecondOrdre(capacite, systeme.constante, amortissement)
    assert sp.simplify(reconstruit.de_bode().expression - capacite / denominateur) == 0
    critique = resistance_critique(inductance, capacite)
    assert sp.simplify(amortissement.subs(resistance, critique)) == 1


def test_les_paires_de_poles_des_planches_19_et_20() -> None:
    """Meme `omega`, meme `sigma`, ou meme `xi` sur un rayon."""

    def depuis(pole: complex) -> SecondOrdre:
        exact = sp.nsimplify(pole)
        return SecondOrdre.depuis_evans(1, -sp.re(exact), sp.Abs(exact))

    lent, rapide = depuis(-0.5 + 1j), depuis(-1 + 1j)
    assert lent.pulsation_propre() == rapide.pulsation_propre() == 1
    assert lent.amortissement < rapide.amortissement
    grave, aigu = depuis(-1 + 1j), depuis(-1 + 3j)
    assert grave.d_evans()[1] == aigu.d_evans()[1] == 1
    assert aigu.pulsation_propre() == 3
    proche, loin = depuis(-1 + 2j), depuis(-3 + 6j)
    assert sp.simplify(proche.amortissement - loin.amortissement) == 0


def test_la_planche_22_a_perdu_le_carre_de_p() -> None:
    """« H(p) = rho^2 / (p + 2 sigma p + rho^2) » : c'est `p^2`.

    Le code de la planche, `den = (1, 2 * xhi, 1)`, est juste ; ses
    courbes culminent a 0,86, 0,71, 0,55 et `1 / e` pour `T = 1`.
    """
    instants = np.arange(0, 10, 0.001)
    sommets = [
        reponse_impulsionnelle([1.0], [1.0, 2 * xi, 1.0], instants).max()
        for xi in (0.1, 0.25, 0.5, 1.0)
    ]
    lus = [0.86, 0.71, 0.55, 0.37]
    assert sommets == pytest.approx(lus, abs=0.005)
    for xi, sommet in zip((0.1, 0.25, 0.5), sommets, strict=False):
        racine = math.sqrt(1 - xi**2)
        assert sommet == pytest.approx(math.exp(-xi * math.acos(xi) / racine), abs=1e-6)
    assert sommets[3] == pytest.approx(math.exp(-1), abs=1e-6)
    demi = SecondOrdre(1, 1, sp.Rational(1, 2))
    symbolique = sp.lambdify(T, impulsionnelle(demi), "numpy")
    assert float(np.max(symbolique(instants))) == pytest.approx(sommets[2], abs=1e-5)
    sans_carre = 1 / (P + 2 * sp.Rational(1, 2) * P + 1)
    assert sp.degree(sp.fraction(sans_carre)[1], P) == 1


def test_la_periode_propre_de_la_planche_26() -> None:
    """`T_0 = 2 pi T / racine(1 - xi^2)`, soit `2 pi T` sans amortissement."""
    assert periode_propre(SecondOrdre(5, 1, 0)) == 2 * sp.pi
    amortie = periode_propre(SecondOrdre(5, 1, sp.Rational(1, 2)))
    assert sp.simplify(amortie - 2 * sp.pi / sp.sqrt(sp.Rational(3, 4))) == 0


def test_la_planche_28_place_les_poles_du_mauvais_cote() -> None:
    """« A(p) = (K / T^2) / (p (p - 1/T_1)(p - 1/T_2)) » : ce sont des `+`.

    La reponse `a(t)` imprimee dessous est la bonne, celle de poles en
    `-1 / T_1` et `-1 / T_2` ; avec les signes de la planche, le systeme
    serait instable.
    """
    systeme = SecondOrdre.depuis_les_constantes_de_temps(5, 2, sp.Rational(1, 2))
    premiere, seconde = 2, sp.Rational(1, 2)
    lente = premiere / (premiere - seconde) * sp.exp(-T / premiere)
    rapide = seconde / (seconde - premiere) * sp.exp(-T / seconde)
    assert sp.simplify(indicielle(systeme) - 5 * (1 - lente - rapide)) == 0
    juste = 5 / ((P + 1 / sp.S(premiere)) * (P + 1 / seconde))
    assert sp.simplify(systeme.de_bode().expression - juste) == 0
    de_la_planche = 5 / ((P - 1 / sp.S(premiere)) * (P - 1 / seconde))
    assert nature_de_la_stabilite(juste) == "strictement stable"
    assert nature_de_la_stabilite(de_la_planche) == "instable"


def test_le_regime_harmonique_du_second_ordre_de_la_planche_32() -> None:
    """`y = |H(j w)| cos(w t + phi)` verifie l'equation du systeme."""
    systeme = SecondOrdre(5, 2, sp.Rational(1, 2))
    analyse = AnalyseHarmonique(systeme.de_bode())
    pulsation = sp.Rational(3, 4)
    module, phase = analyse.module(pulsation), analyse.argument(pulsation)
    regime = module * sp.cos(pulsation * T + phase)
    equation = 4 * sp.diff(regime, T, 2) + 2 * sp.diff(regime, T) + regime
    assert sp.simplify(equation - 5 * sp.cos(pulsation * T)) == 0


def test_un_systeme_d_ordre_trois_est_une_somme_d_ordres_un_et_deux() -> None:
    """Planches 34 et suivantes : un pole reel et une paire conjuguee."""
    transmittance = Transmittance(expression=10 / ((P + 1) * (P**2 + 2 * P + 5)))
    assert est_du_troisieme_ordre(transmittance)
    assert ordre_du_systeme(transmittance) == 3
    termes = sp.Add.make_args(sp.apart(transmittance.expression, P))
    degres = sorted(sp.degree(sp.fraction(terme)[1], P) for terme in termes)
    assert degres == [1, 2]
    formes = {nature_de_la_composante(complex(pole)) for pole in transmittance.poles()}
    assert formes == {"exponentielle decroissante", "sinusoide amortie"}
