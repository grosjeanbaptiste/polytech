"""Signaux et systemes : la feuille « Boucle fermee - stabilite », trois exercices.

Sources : `SignauxSystemes/EXS/Exs_07_Boucle_fermee_Stabilite.pdf` (une
page) et ses reponses finales manuscrites,
`Exs_07_Boucle_fermee_Stabilite_Reponses.pdf`, transcrites dans
`transcriptions/SignauxSystemes/EXS/Exs_07_Boucle_fermee_Stabilite_Reponses.md`.

Chaque reponse est recopiee comme oracle et recoupee par un autre chemin :
Hurwitz contre Nyquist, les racines contre le critere, une simulation
contre le raisonnement en frequence.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from polytech.signaux.application import asservi_exs_07 as un
from polytech.signaux.application import nyquist_exs_07 as deux
from polytech.signaux.application import trois_boucles_exs_07 as trois
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import PULSATION as W

# --- Exercice 1 : un controleur proportionnel -------------------------------


def test_ex1_la_boucle_ouverte() -> None:
    """`K x 1/(p + 1) x 10/(p (2p + 1))`, capteur unitaire."""
    attendue = 10 * un.K / (P * (P + 1) * (2 * P + 1))
    assert sp.cancel(un.boucle().ouverte() - attendue) == 0


def test_ex1_la_transmittance_isochrone_des_reponses() -> None:
    """Page 1 : `a = -30/((1+w^2)(1+4w^2))`, `b = 10 (2w^2-1)/((1+w^2) w (1+4w^2))`."""
    reelle, imaginaire = un.parties_isochrones()
    commun = (1 + W**2) * (1 + 4 * W**2)
    assert sp.simplify(reelle + 30 / commun) == 0
    assert sp.simplify(imaginaire - 10 * (2 * W**2 - 1) / (W * commun)) == 0


def test_ex1_la_courbe_de_nyquist_des_reponses() -> None:
    """Page 1 : asymptote en `-30`, croisement de l'axe reel en `-6,67`."""
    assert un.asymptote() == -30
    pulsation, abscisse = un.croisement()
    assert pulsation == 1 / sp.sqrt(2)
    assert abscisse == -sp.Rational(20, 3)
    assert float(abscisse) == pytest.approx(-6.67, abs=5e-3)


def test_ex1_instable_pour_un_gain_unite() -> None:
    """Page 1 : « instable en boucle fermee pour K = 1 » -- la courbe entoure -1."""
    assert not un.est_stable(1)
    assert un.poles_instables(1) == 2


def test_ex1_le_gain_critique_des_reponses() -> None:
    """Page 1 : « instable pour tout K >= 0,15 »."""
    assert un.gain_critique_par_nyquist() == sp.Rational(3, 20)
    assert un.est_stable(sp.Rational(14, 100))
    assert not un.est_stable(sp.Rational(16, 100))


def test_ex1_hurwitz_rend_la_meme_limite() -> None:
    """Question 3, que les reponses laissent en blanc : `3 - 20 K > 0`."""
    assert un.condition_de_hurwitz() == 3 - 20 * un.K
    assert un.gain_critique_par_hurwitz() == un.gain_critique_par_nyquist()


def test_ex1_a_la_limite_la_boucle_oscille_a_la_pulsation_du_croisement() -> None:
    """Second chemin : pour `K = 0,15`, deux poles en `+-j/racine(2)`."""
    poles = un.poles_en_boucle_fermee(sp.Rational(3, 20))
    imaginaires = sorted(sp.im(pole) for pole in poles if sp.re(pole) == 0)
    assert imaginaires == [-1 / sp.sqrt(2), 1 / sp.sqrt(2)]


def test_ex1_la_boucle_fermee_des_reponses() -> None:
    """Page 1 : `T_R = 10 K/(2p^3 + 3p^2 + p + 10 K)`."""
    attendue = 10 * un.K / (2 * P**3 + 3 * P**2 + P + 10 * un.K)
    assert sp.cancel(un.boucle().vis_a_vis_de_la_consigne() - attendue) == 0


def test_ex1_la_perturbation_a_le_meme_denominateur() -> None:
    """`T_P = -10 (p + 1)/(2p^3 + 3p^2 + p + 10 K)` : meme stabilite."""
    attendue = -10 * (P + 1) / (2 * P**3 + 3 * P**2 + P + 10 * un.K)
    assert sp.cancel(un.boucle().vis_a_vis_de_la_perturbation() - attendue) == 0


# --- Exercice 2 : Nyquist, puis un gain de 20 -------------------------------


def test_ex2_l_asymptote_des_reponses() -> None:
    """Page 1 : la courbe vient du bas le long de la verticale `-3`."""
    assert deux.asymptote() == -3


def test_ex2_la_courbe_ne_coupe_jamais_l_axe_reel() -> None:
    """La partie imaginaire, `-(1 + 10 w^2)/(w (25 w^2 + 1))`, est toujours negative."""
    _, imaginaire = deux.parties_isochrones()
    attendue = -(1 + 10 * W**2) / (W * (25 * W**2 + 1))
    assert sp.simplify(imaginaire - attendue) == 0
    assert deux.coupe_l_axe_reel() is False


def test_ex2_instable_en_boucle_ouverte_au_sens_des_reponses() -> None:
    """Page 2 : un pole a l'origine -- stable au sens large seulement."""
    assert deux.nature_en_boucle_ouverte() == "stable au sens large"


def test_ex2_stable_en_boucle_fermee_pour_tout_gain_positif() -> None:
    """Page 2 : « oui, stable en boucle fermee pour tout K > 0 »."""
    assert (
        deux.equation_caracteristique_de_la_boucle()
        == 5 * P**2 + (1 + 2 * deux.K) * P + deux.K
    )
    for gain in (sp.Rational(1, 100), 1, 20, 1000):
        assert deux.est_stable(gain)


def test_ex2_les_poles_pour_un_gain_de_vingt() -> None:
    """Second chemin : `5p^2 + 41p + 20 = 0`, deux racines reelles negatives."""
    poles = deux.poles_en_boucle_fermee(20)
    assert all(pole.is_real and pole < 0 for pole in poles)
    assert sp.simplify(sp.prod(poles)) == 4
    assert sum(poles) == -sp.Rational(41, 5)


# --- Exercice 3 : trois boucles ---------------------------------------------


def test_ex3_le_tableau_des_reponses() -> None:
    """Page 2 : (a) instable/instable, (b) stable/stable, (c) instable/stable."""
    assert trois.verdicts() == {
        "a": ("instable", "instable"),
        "b": ("stable", "stable"),
        "c": ("instable", "stable"),
    }


def test_ex3a_routh_condamne_la_boucle() -> None:
    """`10p^3 + p^2 + 10 = 0` : le coefficient de `p` manque."""
    assert trois.equation_caracteristique("a") == 10 * P**3 + P**2 + 10


def test_ex3b_le_module_ne_depasse_jamais_un() -> None:
    """`|G| = 1/racine((1+w^2)(1+9w^2))` : le retard n'y change rien."""
    for pulsation in (1e-3, 0.1, 1.0, 10.0):
        assert trois.module_de_b(pulsation) < 1
    assert trois.module_de_b(0.0) == pytest.approx(1.0)


def test_ex3b_le_module_est_celui_de_la_boucle_retard_compris() -> None:
    """Second chemin : `G(j w)` evalue en complexe, avec son `exp(-10 j w)`."""
    for pulsation in (0.3, 1.0, 4.0):
        rationnelle = complex(trois.RATIONNELLES["b"].subs(P, sp.I * pulsation))
        retard = complex(sp.exp(-sp.I * trois.RETARD * pulsation))
        attendu = abs(rationnelle * retard)
        assert trois.module_de_b(pulsation) == pytest.approx(attendu, rel=1e-12)


def test_ex3b_par_la_simulation() -> None:
    """Second chemin : la reponse indicielle de la boucle a retard tend vers 1/2."""
    instants = np.linspace(0.0, 400.0, 8001)
    reponse = trois.indicielle_de_b(instants)
    assert reponse[-1] == pytest.approx(0.5, abs=1e-3)
    assert np.max(np.abs(reponse)) < 1.5


def test_ex3c_les_poles_de_la_boucle_fermee() -> None:
    """`p^2 + p + 10 = 0` : partie reelle `-1/2`."""
    poles = trois.poles_en_boucle_fermee("c")
    assert {sp.re(pole) for pole in poles} == {-sp.Rational(1, 2)}
