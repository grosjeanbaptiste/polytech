"""Signaux et systemes : la feuille « Performances des systemes boucles ».

Sept exercices.

Sources : `SignauxSystemes/EXS/Exs_08_Performances.pdf` (cinq pages) et ses
reponses finales manuscrites, `Exs_08_Performances_Reponses.pdf`,
transcrites dans
`transcriptions/SignauxSystemes/EXS/Exs_08_Performances_Reponses.md`.

Les reponses officielles sont recopiees comme oracles. Celles des exercices
4 a 7 sont des **lectures d'abaque** : la bibliotheque calcule les memes
grandeurs sur la reponse frequentielle exacte, et la tolerance des tests
est celle d'une lecture. Une simulation de la boucle fermee, retard
compris, dit ensuite ce que valent les regles d'estimation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from polytech.signaux.application import black_exs_08 as bk
from polytech.signaux.application import gains_exs_08 as ga
from polytech.signaux.application import marges_exs_08 as mg
from polytech.signaux.application import regimes_exs_08 as rg

# --- Exercice 1 : un procede integrateur ------------------------------------


def test_ex1_la_consigne() -> None:
    """Page 1 : `y = 1`, `v = 0`, `u = 0`, `eps = 0` pour une consigne de 2."""
    assert rg.regime_un(consigne=2, perturbation=0) == rg.Regime(y=1, v=0, u=0, ecart=0)


def test_ex1_la_perturbation() -> None:
    """Page 1 : `y = -1,25`, `v = 5`, `u = 2,5 = eps` pour une perturbation de 5."""
    attendu = rg.Regime(
        y=-sp.Rational(5, 4), v=5, u=sp.Rational(5, 2), ecart=sp.Rational(5, 2)
    )
    assert rg.regime_un(consigne=0, perturbation=5) == attendu


def test_ex1_par_le_theoreme_de_la_valeur_finale() -> None:
    """Second chemin : `lim p Y(p)`, les transmittances de la boucle entiere."""
    assert rg.sortie_finale_un(consigne=2, perturbation=0) == 1
    assert rg.sortie_finale_un(consigne=0, perturbation=5) == -sp.Rational(5, 4)


def test_ex1_sans_correction_la_boucle_est_instable() -> None:
    """`alpha = beta` : `30p^3 + 13p^2 + p + 20`, et `13 < 600`."""
    assert rg.est_stable_un(alpha=1, beta=1) is False
    assert rg.est_stable_un(alpha=5, beta=sp.Rational(1, 100)) is True


# --- Exercice 2 : un correcteur a action integrale --------------------------


@pytest.mark.parametrize(
    ("consigne", "perturbation", "y", "u"),
    [(2, 0, 1, 0.2), (0, 10, 0, 2.0), (2, 10, 1, 2.2)],
)
def test_ex2_les_trois_cas(consigne: int, perturbation: int, y: int, u: float) -> None:
    """Page 1 : les trois lignes, `eps = 0` partout."""
    regime = rg.regime_deux(consigne=consigne, perturbation=perturbation)
    assert regime.y == y
    assert float(regime.u) == pytest.approx(u, rel=1e-12)
    assert regime.ecart == 0


def test_ex2_le_troisieme_cas_est_la_somme_des_deux_premiers() -> None:
    """Second chemin : la superposition, la boucle etant lineaire."""
    premier = rg.regime_deux(consigne=2, perturbation=0)
    deuxieme = rg.regime_deux(consigne=0, perturbation=10)
    troisieme = rg.regime_deux(consigne=2, perturbation=10)
    assert troisieme.y == premier.y + deuxieme.y
    assert troisieme.u == premier.u + deuxieme.u


def test_ex2_la_boucle_est_stable_malgre_son_retard() -> None:
    """Sans quoi le regime n'existerait pas : la marge de phase est positive."""
    assert rg.marge_de_phase_deux() > 0
    assert rg.marge_de_phase_deux() == pytest.approx(59.7, abs=0.1)


# --- Exercice 3 : un controleur proportionnel et un retard ------------------


def test_ex3_la_marge_de_phase() -> None:
    """Page 1 : `20,8` degres pour `Gc = 1`, au module un en `racine(3)`."""
    marge, pulsation = mg.marge_de_phase(1.0)
    assert pulsation == pytest.approx(math.sqrt(3), rel=1e-9)
    assert marge == pytest.approx(20.8, abs=0.05)


def test_ex3_la_marge_par_la_formule() -> None:
    """Second chemin : `180 - racine(3) rad - arctan(racine(3))`."""
    attendue = 180 - math.degrees(math.sqrt(3)) - 60
    assert mg.marge_de_phase(1.0)[0] == pytest.approx(attendue, abs=1e-6)


def test_ex3_le_gain_pour_soixante_degres() -> None:
    """Page 1 : `Gc = 0,786`."""
    gain = mg.gain_pour_la_marge(60.0)
    assert gain == pytest.approx(0.786, abs=5e-4)
    assert mg.marge_de_phase(gain)[0] == pytest.approx(60.0, abs=1e-6)


def test_ex3_la_sortie_en_regime() -> None:
    """Page 1 : `1,5 + (-0,75) + (-2,5) = -1,75`."""
    ideale, de_consigne, de_perturbation = mg.decomposition()
    assert (ideale, de_consigne, de_perturbation) == (
        sp.Rational(3, 2),
        -sp.Rational(3, 4),
        -sp.Rational(5, 2),
    )
    assert mg.sortie_finale() == -sp.Rational(7, 4)


# --- Exercice 4 : lire la fonction M sur l'abaque ---------------------------


def test_ex4_les_lectures_officielles() -> None:
    """Page 1 : `-3` dB, `10` dB, `0,45` rad/s."""
    m = bk.caracteristiques("quatre")
    assert m.en_continu_en_decibels() == pytest.approx(-3.0, abs=0.1)
    assert m.maximum_en_decibels() == pytest.approx(10.0, abs=0.1)
    assert m.bande_passante == pytest.approx(0.45, abs=0.01)


def test_ex4_le_module_en_continu_se_calcule() -> None:
    """Second chemin : `L(0) = 0,2 x 2,4/0,2 = 2,4`, donc `|M(0)| = 2,4/3,4`."""
    m = bk.caracteristiques("quatre")
    assert m.en_continu == pytest.approx(2.4 / 3.4, rel=1e-6)


def test_ex4_l_estimation_officielle() -> None:
    """Page 1 : `y = 0,707`, `D = 87 %`, `y_max = 1,32`, `4,2 < t_m < 6,3` s."""
    estimation = bk.estimation("quatre")
    assert estimation.finale == pytest.approx(0.707, abs=2e-3)
    assert estimation.depassement == pytest.approx(87.0, abs=1.0)
    assert estimation.maximum == pytest.approx(1.32, abs=0.01)
    assert estimation.montee == pytest.approx((4.2, 6.3), abs=0.15)


def test_ex4_la_simulation_dit_ce_que_vaut_l_estimation() -> None:
    """La valeur finale est exacte ; le maximum reel est sous le maximum estime."""
    finale, maximum, depart = bk.simulation("quatre")
    estimation = bk.estimation("quatre")
    assert finale == pytest.approx(estimation.finale, abs=2e-3)
    assert finale < maximum < estimation.maximum
    assert depart == pytest.approx(5.0, abs=0.1)


# --- Exercice 5 : tracer la courbe de Black, puis regler K ------------------


@pytest.mark.parametrize(
    ("pulsation", "argument", "gain"),
    [
        (0.1, -46.6, -0.98),
        (0.25, -101.4, -4.15),
        (0.35, -130.34, -6.22),
        (0.5, -168.2, -8.87),
        (0.6, -191.4, -10.37),
        (0.7, -213.7, -11.72),
    ],
)
def test_ex5_le_tableau_officiel(
    pulsation: float, argument: float, gain: float
) -> None:
    """Page 2 : les six points de la courbe de Black pour `K = 1`."""
    obtenu = bk.point_de_black(pulsation)
    assert obtenu[0] == pytest.approx(argument, abs=0.06)
    assert obtenu[1] == pytest.approx(gain, abs=0.01)


def test_ex5_les_marges_pour_un_gain_unite() -> None:
    """Page 2 : `MG = 9,7` dB ; la marge de phase vaut 180 degres, lue en `w = 0`."""
    assert bk.marge_de_gain_en_decibels(1.0) == pytest.approx(9.7, abs=0.06)
    assert bk.module_maximal_de_la_boucle(1.0) == pytest.approx(1.0, rel=1e-9)


def test_ex5_doubler_le_gain_retire_six_decibels_de_marge() -> None:
    """Un gain translate la courbe : `20 log 2` de marge en moins, un module double."""
    ecart = bk.marge_de_gain_en_decibels(1.0) - bk.marge_de_gain_en_decibels(2.0)
    assert ecart == pytest.approx(20 * math.log10(2), rel=1e-9)
    assert bk.module_maximal_de_la_boucle(2.0) == pytest.approx(2.0, rel=1e-9)


def test_ex5_le_gain_pour_quatre_vingt_dix_degres() -> None:
    """Page 2 : `K = 1,5`."""
    assert bk.gain_pour_la_marge(90.0) == pytest.approx(1.5, abs=0.03)


def test_ex5_les_lectures_pour_un_gain_de_deux() -> None:
    """Page 3 : `-3,5` dB, environ `6` dB, `0,8` rad/s, a une lecture pres."""
    m = bk.caracteristiques("cinq")
    assert m.en_continu_en_decibels() == pytest.approx(-3.5, abs=0.05)
    assert m.maximum_en_decibels() == pytest.approx(6.0, abs=1.0)
    assert m.bande_passante == pytest.approx(0.8, abs=0.05)


def test_ex5_l_estimation_officielle_reprise_sur_ses_propres_lectures() -> None:
    """Page 3 : `9 + 6 x 9,5 = 66 %`, `y = 0,67`, `y_max = 1,1`, `2,4 < t_m < 3,6` s."""
    estimation = bk.estimer(en_continu_db=-3.5, maximum_db=6.0, bande=0.8)
    assert estimation.depassement == pytest.approx(66.0, abs=1e-9)
    assert estimation.finale == pytest.approx(0.67, abs=5e-3)
    assert estimation.maximum == pytest.approx(1.1, abs=0.02)
    assert estimation.montee == pytest.approx((2.4, 3.6), abs=0.07)


def test_ex5_la_simulation() -> None:
    """Second chemin : la boucle fermee simulee demarre a 3 s et culmine pres de 1,1."""
    finale, maximum, depart = bk.simulation("cinq")
    assert finale == pytest.approx(2 / 3, abs=2e-3)
    assert maximum == pytest.approx(1.1, abs=0.02)
    assert depart == pytest.approx(3.0, abs=0.1)


# --- Exercice 6 : une courbe donnee, un controleur inconnu ------------------


def test_ex6_l_estimation_officielle() -> None:
    """Page 3 : `0` dB, `2,3` dB, `0,24` rad/s ; d'ou `22,8 %`, `7,9 < t_m < 11,8` s."""
    estimation = ga.estimation_six()
    assert estimation.finale == pytest.approx(1.0)
    assert estimation.depassement == pytest.approx(22.8, abs=1e-9)
    assert estimation.maximum == pytest.approx(1.23, abs=5e-3)
    assert estimation.montee == pytest.approx((7.9, 11.8), abs=0.06)


# --- Exercice 7 : regler K sur la courbe de Black ---------------------------


def test_ex7_le_gain_limite() -> None:
    """Page 4 : stable pour `K < 1,55`, la marge de gain valant `3,8` dB."""
    marge, limite = ga.marge_et_gain_limite()
    assert marge == pytest.approx(3.8, abs=0.05)
    assert limite == pytest.approx(1.55, abs=0.01)
    assert limite == pytest.approx(10 ** (marge / 20), rel=1e-12)


def test_ex7_la_limite_fait_passer_la_courbe_par_le_point_critique() -> None:
    """Second chemin : pour ce gain, `K G(j w)` vaut `-1` a la pulsation de la marge."""
    valeur = ga.boucle_a_la_limite()
    assert valeur.real == pytest.approx(-1.0, abs=1e-6)
    assert valeur.imag == pytest.approx(0.0, abs=1e-6)


def test_ex7_le_gain_pour_le_depassement_demande() -> None:
    """Entre 20 et 25 % : la regle `9 + 6 ecart` y mene pour `K` voisin de 0,5.

    Les reponses officielles ne donnent pas ce gain ; c'est celui que
    l'enonce reprend aux deux questions suivantes.
    """
    gain = ga.gain_pour_le_depassement(22.5)
    assert gain == pytest.approx(0.5, abs=0.05)
    assert 20.0 <= ga.estimation_sept(0.5).depassement <= 25.0


def test_ex7_les_valeurs_finales() -> None:
    """Page 4 : `2,5` pour une consigne de 5, `-1` pour une perturbation de 2."""
    assert ga.sortie_finale(consigne=5, perturbation=0) == sp.Rational(5, 2)
    assert ga.sortie_finale(consigne=0, perturbation=2) == -1


def test_ex7_la_reponse_officielle_ecrit_une_consigne_de_un_demi() -> None:
    """Page 4 : « R(p) = 0,5/p -> 2,5 ». Avec 0,5 la sortie vaudrait 0,25."""
    assert ga.sortie_finale(consigne=sp.Rational(1, 2), perturbation=0) == sp.Rational(
        1, 4
    )


def test_ex7_la_simulation_confirme_la_stabilite_de_part_et_d_autre() -> None:
    """Second chemin : sous la limite la reponse se pose, au-dessus elle diverge."""
    instants = np.linspace(0.0, 1500.0, 15001)
    assert ga.amplitude_finale(1.3, instants) < 0.05
    assert ga.amplitude_finale(1.8, instants) > 1.0
