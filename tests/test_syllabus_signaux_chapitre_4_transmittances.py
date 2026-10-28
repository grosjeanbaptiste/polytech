"""Le syllabus de signaux et systemes, chapitre 4 : les transmittances.

Source : `SignauxSystemes/SYLLABUS/Syllabus_2019_Chapitre_04_Transmittances.pdf`
(31 pages), sections 4.1 a 4.4, lues sur les pages rendues.

Les oracles : l'exemple 4.1 (le ralenti audio), (4.26) et (4.28) par la
regle de Mason, le graphe de l'exemple 4.5, les montages (4.30) a (4.35)
et la transmittance de l'exemple 4.6.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from polytech.electronique.domain.laplace import (
    diviseur_de_tension,
    impedance_de_bobine,
    impedance_de_condensateur,
    impedance_de_resistance,
    impedances_en_parallele,
    impedances_en_serie,
)
from polytech.electronique.domain.montages import (
    integrateur,
    somme_ponderee,
    transfert_a_impedances,
)
from polytech.maths.domain.comparaison import (
    egales,
    proches,
    proches_relativement,
    suite_nulle,
    suites_proches,
)
from polytech.signaux.domain.echantillons import reponse_par_superposition
from polytech.signaux.domain.fluence import (
    coefficients_normalises,
    regle_de_mason,
    transmittance_du_graphe,
)
from polytech.signaux.domain.forme_generale import (
    forme_factorisee,
    mode_d_une_paire_conjuguee,
)
from polytech.signaux.domain.heaviside import original, residu
from polytech.signaux.domain.modeles import transmittance_d_une_equation
from polytech.signaux.domain.stabilite_des_poles import nature_de_la_stabilite
from polytech.signaux.domain.symboles import IMAGE, TEMPS
from polytech.signaux.domain.systemes import Systeme, convolution
from polytech.signaux.domain.temporel import ReponsesTemporelles
from polytech.signaux.domain.transmittance import Transmittance

P, T = IMAGE, TEMPS
R, L, C = sp.symbols("R L C", positive=True)


def test_l_exemple_4_1_le_ralenti_audio_est_lineaire_mais_pas_invariant() -> None:
    """`y(t) = x(a t)` : le retard est ralenti lui aussi."""
    ralenti = Systeme(operateur=lambda entree: entree.subs(T, T / 2))
    assert ralenti.est_lineaire()
    assert not ralenti.est_invariant()
    integrateur_pur = Systeme(operateur=lambda entree: sp.diff(entree, T))
    assert integrateur_pur.est_invariant()


def test_l_exponentielle_complexe_est_fonction_propre_d_un_sli() -> None:
    """(4.3) : a l'entree `exp(p t)` repond `H(p) exp(p t)`."""
    transmittance = transmittance_d_une_equation(sortie=(2, 3, 5), entree=(1, 4))
    valeur = sp.Rational(-1, 2) + 3 * sp.I
    entree = sp.exp(valeur * T)
    sortie = transmittance.expression.subs(P, valeur) * entree
    gauche = 2 * sp.diff(sortie, T, 2) + 3 * sp.diff(sortie, T) + 5 * sortie
    droite = sp.diff(entree, T) + 4 * entree
    assert egales(gauche, droite)


def test_la_forme_generale_de_h_p_est_une_fraction_rationnelle() -> None:
    """(4.8) a (4.10) : les coefficients de l'equation sont ceux de `H(p)`."""
    transmittance = transmittance_d_une_equation(sortie=(2, 3, 5), entree=(1, 4))
    assert egales(transmittance.expression, (P + 4) / (2 * P**2 + 3 * P + 5))


def test_les_poles_les_zeros_et_un_facteur_definissent_h_p() -> None:
    """(4.11) : `K produit(p - z_i) / produit(p - p_i)`."""
    factorisee = forme_factorisee(3, zeros=(-4,), poles=(-1, -2))
    assert egales(factorisee.expression, 3 * (P + 4) / ((P + 1) * (P + 2)))
    assert factorisee.poles() == (-2, -1)
    assert factorisee.zeros() == (-4,)
    with pytest.raises(ValueError, match="aucun pole"):
        forme_factorisee(3, zeros=(-4,), poles=())


def test_la_reponse_indicielle_est_l_integrale_de_la_reponse_impulsionnelle() -> None:
    """(4.13) : `H(p) / p <-> a(t)`."""
    reponses = ReponsesTemporelles(Transmittance(expression=1 / (P + 2)))
    integrale = sp.integrate(reponses.impulsionnelle(), (T, 0, T))
    assert egales(integrale, reponses.indicielle())


def test_la_reponse_est_la_convolution_de_l_entree_et_de_h() -> None:
    """(4.18), et (4.5) pour le meme calcul en `p`."""
    impulsionnelle = sp.exp(-2 * T)
    par_convolution = convolution(sp.Integer(1), impulsionnelle)
    par_laplace = original(1 / (P * (P + 2)))
    assert egales(par_convolution, par_laplace)


def test_la_reponse_somme_des_reponses_impulsionnelles_decalees() -> None:
    """(4.17) : `y(t) ~ somme x(t_k) h(t - t_k) Delta`, avec `Delta` petit."""
    pas = 0.001
    instants = np.arange(0, 3, pas)
    sortie = reponse_par_superposition(
        np.ones_like(instants), np.exp(-2 * instants), pas
    )
    attendue = (1 - np.exp(-2 * instants)) / 2
    assert suites_proches(sortie[: len(instants)], attendue, 2 * pas)
    with pytest.raises(ValueError, match="pas nul ou negatif"):
        reponse_par_superposition(np.ones(3), np.ones(3), 0.0)


def test_la_forme_generale_de_h_t_pour_une_paire_de_poles_conjugues() -> None:
    """(4.21) : `2 |A| exp(-sigma t) cos(omega t + arg A)`, sur l'exemple 2.14."""
    instants = np.array([0.0, 0.3, 1.1, 2.7])
    amorti = (P + 1) / ((P + 3) ** 2 + 16)
    for image, pole in (((P + 1) / (P**2 + 4), 2 * sp.I), (amorti, -3 + 4 * sp.I)):
        mode = mode_d_une_paire_conjuguee(residu(image, pole), pole)
        ecart = sp.lambdify(T, mode - original(image), "numpy")
        assert suite_nulle(ecart(instants), 1e-12)
    pur = mode_d_une_paire_conjuguee(residu((P + 1) / (P**2 + 4), 2 * sp.I), 2 * sp.I)
    assert proches_relativement(float(pur.subs(T, 0)), 1.0, 1e-6)
    assert proches(
        float(sp.Abs(residu((P + 1) / (P**2 + 4), 2 * sp.I)) * 2), 1.1180, 1e-4
    )


def test_la_stabilite_se_lit_sur_les_poles() -> None:
    """4.1.8 : stricte a gauche de l'axe, large si l'axe ne porte que des simples."""
    assert nature_de_la_stabilite(1 / (P + 1)) == "strictement stable"
    assert nature_de_la_stabilite(1 / (P**2 + 4)) == "stable au sens large"
    assert nature_de_la_stabilite(1 / P) == "stable au sens large"
    assert nature_de_la_stabilite(1 / (P**2 + 4) ** 2) == "instable"
    assert nature_de_la_stabilite(1 / P**2) == "instable"
    assert nature_de_la_stabilite(1 / (P - 1)) == "instable"


def test_l_exemple_4_3_la_boucle_fermee_par_la_regle_de_mason() -> None:
    """(4.25) : un parcours `K G`, une boucle `-K G H`.

    Le syllabus imprime (4.26) avec un signe moins devant la fraction et
    un `k` minuscule ; la regle qu'il vient d'enoncer ne donne ni l'un ni
    l'autre.
    """
    gain, directe, retour = sp.symbols("K G H")
    fermee = regle_de_mason([gain * directe], [-gain * directe * retour])
    assert egales(fermee, gain * directe / (1 + gain * directe * retour))
    assert not egales(fermee, -gain * directe / (1 + gain * directe * retour))
    with pytest.raises(ValueError, match="aucun parcours"):
        regle_de_mason([], [retour])


def test_l_exemple_4_4_le_circuit_par_son_graphe_de_fluence() -> None:
    """(4.28) : parcours `p L + R`, boucle `-p C (p L + R)`."""
    par_le_graphe = regle_de_mason([P * L + R], [-P * C * (P * L + R)])
    imprimee = (P * L + R) / (P**2 * L * C + P * R * C + 1)
    assert egales(par_le_graphe, imprimee)
    branche = impedances_en_serie(
        impedance_de_bobine(L, P), impedance_de_resistance(R, P)
    )
    par_les_impedances = impedances_en_parallele(
        branche, impedance_de_condensateur(C, P)
    )
    assert egales(par_les_impedances, imprimee)


def test_l_exemple_4_5_le_graphe_d_une_transmittance_du_troisieme_ordre() -> None:
    """(4.29) : retours -20, -200, -1000 ; gains directs 1, 0, 100."""
    transmittance = (P**2 + 100) / (P**3 + 20 * P**2 + 200 * P + 1000)
    alphas, betas = coefficients_normalises(transmittance)
    assert alphas == (20, 200, 1000)
    assert betas == (1, 0, 100)
    du_graphe = transmittance_du_graphe(retours=(-20, -200, -1000), directs=(1, 0, 100))
    assert egales(du_graphe, transmittance)


def test_la_normalisation_divise_par_le_coefficient_de_tete() -> None:
    """(4.22) : `alpha_i = a_i / a_n`, `beta_i = b_i / a_n`."""
    alphas, betas = coefficients_normalises((4 * P + 6) / (2 * P**2 + 8 * P + 10))
    assert alphas == (4, 5)
    assert betas == (2, 3)
    with pytest.raises(ValueError, match="degre du numerateur"):
        coefficients_normalises(P**2 / (P**2 + 1))
    with pytest.raises(ValueError, match="autant de retours que de gains directs"):
        transmittance_du_graphe(retours=(-1,), directs=(1, 2))


def test_un_coefficient_de_tete_different_de_un_divise_tout() -> None:
    demi = sp.Rational(1, 2)
    alphas, betas = coefficients_normalises((P + 1) / (2 * P**2 + 3 * P + 5))
    assert alphas == (3 * demi, 5 * demi)
    assert betas == (demi, demi)


def test_l_exercice_4_6_le_graphe_de_la_capsule_spatiale() -> None:
    """« Proposer un graphe de fluence » : trois integrateurs, un seul gain direct."""
    transmittance = 5 / (P**3 + P**2 + 10 * P + 15)
    alphas, betas = coefficients_normalises(transmittance)
    assert alphas == (1, 10, 15)
    assert betas == (0, 0, 5)
    retours = tuple(-alpha for alpha in alphas)
    du_graphe = transmittance_du_graphe(retours=retours, directs=betas)
    assert egales(du_graphe, transmittance)


def test_la_structure_elementaire_a_amplificateur_operationnel() -> None:
    """(4.30) a (4.33) : `H = -Z_2 / Z_1`, l'inverseur et l'integrateur."""
    r1, r2 = sp.symbols("R_1 R_2", positive=True)
    assert transfert_a_impedances(r1, r2) == -r2 / r1
    capacite = impedance_de_condensateur(C, P)
    assert egales(transfert_a_impedances(R, capacite), -(1 / (P * R * C)))
    assert transfert_a_impedances(R, capacite).subs({R: 100, C: 2}) == integrateur(
        100, 2, P
    )
    with pytest.raises(ValueError, match="impedance d'entree nulle"):
        transfert_a_impedances(0, r2)


def test_l_additionneur_pondere_par_r_sur_r_i() -> None:
    """Le syllabus imprime (4.34) et (4.35) a l'envers : `-R_i / R`.

    Sa figure 4.25 place `R_i` sur l'entree et `R` en contre-reaction ; sa
    propre formule (4.30) donne alors `-R / R_i`.
    """
    contre_reaction, voie = 10e3, 2e3
    par_la_structure = transfert_a_impedances(voie, contre_reaction)
    assert proches_relativement(par_la_structure, -5.0, 1e-6)
    ponderee = somme_ponderee(contre_reaction, (voie,))
    assert len(ponderee) == 1
    assert proches_relativement(ponderee[0], -5.0, 1e-6)
    assert not proches_relativement(par_la_structure, -voie / contre_reaction, 1e-6)


def test_l_exemple_4_6_la_transmittance_d_un_circuit_passif() -> None:
    """`H = R / (p^2 R L C + p (R^2 C + L) + 2 R)`."""
    charge = impedances_en_serie(
        impedance_de_bobine(L, P), impedance_de_resistance(R, P)
    )
    noeud = impedances_en_parallele(impedance_de_condensateur(C, P), charge)
    au_noeud = diviseur_de_tension(impedance_de_resistance(R, P), noeud)
    sortie = au_noeud * diviseur_de_tension(impedance_de_bobine(L, P), R)
    imprimee = R / (P**2 * R * L * C + P * (R**2 * C + L) + 2 * R)
    assert egales(sortie, imprimee)


def test_l_exemple_4_6_par_les_lois_de_kirchhoff_du_syllabus() -> None:
    """Le systeme de trois equations de la page 22, resolu tel qu'il est ecrit."""
    entree = sp.Symbol("U_i")
    matrice = sp.Matrix([[R, 0, 1 / (P * C)], [0, P * L + R, 1 / (P * C)], [1, 1, -1]])
    _, dans_la_bobine, _ = matrice.solve(sp.Matrix([entree, 0, 0]))
    imprime = -entree / (P**2 * R * L * C + P * (R**2 * C + L) + 2 * R)
    assert egales(dans_la_bobine, imprime)
