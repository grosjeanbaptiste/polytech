"""Signaux et systemes : la feuille « Analyse harmonique », six exercices.

Source : `SignauxSystemes/EXS/Exs_06_Analyse_harmonique.pdf`, trois pages,
sans aucune reponse. Les oracles sont donc ce que la feuille imprime -- les
pulsations et les pentes portees sur ses courbes -- et des valeurs
calculees a la main, chacune recoupee par un autre chemin : la reponse
frequentielle exacte, les marges de la bibliotheque, `scipy.signal`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp
from scipy import signal

from polytech.signaux.application import black_exs_06 as bl
from polytech.signaux.application import bode_exs_06 as bo
from polytech.signaux.application import second_ordre_exs_06 as so
from polytech.signaux.domain.symboles import IMAGE as P

# --- Exercice 1 : associer cinq courbes a cinq transmittances ---------------


def test_ex1_les_pentes_des_cinq_traces() -> None:
    """Les suites de pentes, en dB par decade, lues sur les formes de Bode."""
    pentes = {nom: bo.trace(nom).pentes() for nom in "abcde"}
    assert pentes == {
        "a": (0, -20, -40),
        "b": (-20, 0, -20),
        "c": (20, 0, -20, 0, -20),
        "d": (-20, 0, -20, -40, -60),
        "e": (20, 0, -20),
    }


def test_ex1_l_association() -> None:
    """Chaque courbe a une suite de pentes qu'une seule transmittance possede."""
    assert bo.association() == {"a": 1, "b": 3, "c": 5, "d": 4, "e": 2}


@pytest.mark.parametrize(("nom", "pulsation"), [("e", 0.125), ("d", 2.0)])
def test_ex1_les_passages_a_zero_decibel_imprimes(nom: str, pulsation: float) -> None:
    """La feuille gradue ses axes en 0,125 et en 2 : l'asymptote y coupe 0 dB."""
    assert bo.trace(nom).gain_en_decibels(pulsation) == pytest.approx(0.0, abs=1e-9)


def test_ex1_le_prolongement_pointille_de_la_cinquieme_courbe() -> None:
    """Gradue en 0,25 : la premiere asymptote, `20 log(4 w)`, prolongee jusqu'a 0 dB.

    La courbe elle-meme casse avant, en 0,2 : son palier reste sous l'axe.
    """
    trace = bo.trace("c")
    assert 1 / trace.gain == pytest.approx(0.25, rel=1e-12)
    pulsation, saut = trace.cassures[0]
    assert pulsation == pytest.approx(0.2)
    assert saut == -1
    assert trace.gain_en_decibels(0.5) == pytest.approx(-1.94, abs=5e-3)


def test_ex1_le_palier_de_la_premiere_courbe() -> None:
    """La courbe (1) porte « 20 » sur son axe : `20 log 10`."""
    assert bo.trace("a").gain_en_decibels(1.0) == pytest.approx(20.0, abs=1e-12)


@pytest.mark.parametrize("nom", list("abcde"))
def test_ex1_loin_des_cassures_l_asymptote_rejoint_la_courbe_reelle(nom: str) -> None:
    """Second chemin : le module exact, deux decades avant et apres les cassures."""
    cassures = [w for w, _ in bo.trace(nom).cassures]
    for pulsation in (min(cassures) / 100, max(cassures) * 100):
        exact = bo.gain_reel(bo.FONCTIONS[nom], pulsation)
        asymptote = bo.trace(nom).gain_en_decibels(pulsation)
        assert exact == pytest.approx(asymptote, abs=0.02)


def test_ex1_aucun_pole_ni_zero_n_est_a_droite() -> None:
    """Le module seul ne distingue pas `1 + p/3` de `1 - p/3` : on le verifie a part."""
    for nom in "abcde":
        numerateur, denominateur = sp.fraction(sp.cancel(bo.FONCTIONS[nom]))
        racines = [*sp.roots(numerateur), *sp.roots(denominateur)]
        assert all(sp.re(racine) <= 0 for racine in racines), nom


# --- Exercice 2 : Bode et Nyquist de 10 (p + 1)/(p (p + 5)) -----------------


def test_ex2_la_forme_de_bode() -> None:
    """`2 (1 + p)/(p (1 + p/5))` : gain 2, un integrateur, cassures en 1 et 5."""
    trace = bo.trace("deux")
    assert (trace.gain, trace.classe) == (2, 1)
    assert trace.cassures == ((1.0, 1), (5.0, -1))


def test_ex2_l_argument_culmine_entre_les_deux_cassures() -> None:
    """Maximum en `racine(5)`, moyenne geometrique : `-90 + arctan(2/racine(5))`."""
    pulsation, argument = bo.sommet_de_l_argument()
    assert pulsation == sp.sqrt(5)
    attendu = -90 + math.degrees(math.atan(2 / math.sqrt(5)))
    assert float(argument) == pytest.approx(attendu, abs=1e-9)
    assert float(argument) == pytest.approx(-48.19, abs=5e-3)


def test_ex2_l_asymptote_verticale_de_nyquist() -> None:
    """Quand `w` tend vers zero, la partie reelle tend vers `8/5`."""
    assert bo.asymptote_de_nyquist() == sp.Rational(8, 5)


def test_ex2_la_courbe_de_nyquist_reste_dans_le_quatrieme_quadrant() -> None:
    """Partie reelle positive, partie imaginaire negative, a toute pulsation."""
    for pulsation in (0.01, 0.5, 2.0, 5.0, 100.0):
        reelle, imaginaire = bo.point_de_nyquist(pulsation)
        assert reelle > 0
        assert imaginaire < 0


def test_ex2_par_scipy() -> None:
    """Second chemin : `scipy.signal.bode` sur les memes coefficients."""
    pulsations = np.array([0.1, 1.0, math.sqrt(5), 5.0, 50.0])
    _, gains, phases = signal.bode(([10, 10], [1, 5, 0]), pulsations)
    for pulsation, gain, phase in zip(pulsations, gains, phases, strict=True):
        assert bo.gain_reel(bo.FONCTIONS["deux"], pulsation) == pytest.approx(gain)
        assert bo.argument_reel(bo.FONCTIONS["deux"], pulsation) == pytest.approx(phase)


# --- Exercice 3 : Black-Nichols de 5/(p (1 + 0,5 p)) ------------------------


def test_ex3_le_passage_a_zero_decibel() -> None:
    """`w^2 (1 + w^2/4) = 25` : `w = 2,86` rad/s, argument `-145,1` degres."""
    pulsation, argument = bl.passage_a_zero_decibel()
    assert pulsation**2 * (1 + pulsation**2 / 4) == pytest.approx(25.0, rel=1e-9)
    assert pulsation == pytest.approx(2.863, abs=5e-4)
    assert argument == pytest.approx(-145.07, abs=5e-3)


def test_ex3_les_marges() -> None:
    """Second chemin : la marge de phase de la bibliotheque, `180 - 145,1`."""
    marges = bl.marges()
    _, argument = bl.passage_a_zero_decibel()
    assert marges.de_phase == pytest.approx(180 + argument, abs=1e-3)
    assert math.isinf(marges.de_gain)


def test_ex3_la_courbe_va_d_un_quart_de_tour_a_un_demi_tour() -> None:
    """De -90 degres aux basses pulsations a -180 aux hautes, sans remonter."""
    arguments = [bl.point_de_black(bl.TROIS, w)[0] for w in (1e-3, 1.0, 10.0, 1e3)]
    assert arguments == sorted(arguments, reverse=True)
    assert arguments[0] == pytest.approx(-90.0, abs=0.1)
    assert arguments[-1] == pytest.approx(-180.0, abs=0.2)


# --- Exercice 4 : Bode de 10 p/((20 p + 1)(p + 0,5)) ------------------------


def test_ex4_la_forme_de_bode() -> None:
    """`20 p/((1 + 20 p)(1 + 2 p))` : un derivateur, cassures en 0,05 et 0,5."""
    trace = bo.trace("quatre")
    assert (trace.gain, trace.classe) == (20, -1)
    assert trace.cassures == ((0.05, -1), (0.5, -1))
    assert trace.pentes() == (20, 0, -20)


def test_ex4_le_palier_est_a_zero_decibel() -> None:
    assert bo.trace("quatre").gain_en_decibels(0.1) == pytest.approx(0.0, abs=1e-9)


def test_ex4_la_courbe_reelle_culmine_sous_le_palier() -> None:
    """Maximum en `racine(0,05 x 0,5)`, ou `|H| = 10/11` : `-0,83` dB."""
    pulsation, gain = bo.sommet_du_gain()
    assert pulsation == pytest.approx(math.sqrt(0.025), rel=1e-6)
    assert gain == pytest.approx(20 * math.log10(10 / 11), abs=1e-6)


def test_ex4_l_ecart_aux_cassures() -> None:
    """A chaque cassure, la courbe reelle passe `3,05` dB sous l'asymptote."""
    for cassure in (0.05, 0.5):
        ecart = bo.trace("quatre").gain_en_decibels(cassure) - bo.gain_reel(
            bo.FONCTIONS["quatre"], cassure
        )
        assert ecart == pytest.approx(3.05, abs=0.01)


# --- Exercice 5 : associer trois courbes de Black ---------------------------


def test_ex5_les_extremes_de_l_argument() -> None:
    """`T1` plonge sous -90, `T3` remonte au-dessus, `T2` part de -180."""
    assert bl.extreme_de_l_argument("T1") == pytest.approx(-115.38, abs=5e-3)
    assert bl.extreme_de_l_argument("T3") == pytest.approx(-53.13, abs=5e-3)
    assert bl.extreme_de_l_argument("T2") == pytest.approx(-125.10, abs=5e-3)


def test_ex5_l_association() -> None:
    """(A) depasse -90 vers la droite, (B) vient de -180, (C) creuse vers -115."""
    assert bl.association() == {"T1": "C", "T2": "B", "T3": "A"}


def test_ex5_les_arguments_aux_deux_bouts() -> None:
    """Second chemin : `T1` et `T3` partent de -90 et y reviennent, `T2` de -180."""
    for nom, bout in (("T1", -90.0), ("T3", -90.0), ("T2", -180.0)):
        for pulsation in (1e-4, 1e4):
            argument, _ = bl.point_de_black(bl.FONCTIONS[nom], pulsation)
            assert argument == pytest.approx(bout, abs=0.3)


# --- Exercice 6 : un second ordre resonant ----------------------------------


def test_ex6_les_parametres() -> None:
    """`10/(p^2 + 0,8 p + 1)` : `K = 10`, `w_n = 1`, `xi = 0,4`."""
    assert so.parametres() == (10, 1, sp.Rational(2, 5))


def test_ex6_la_resonance() -> None:
    """`w_r = racine(1 - 2 xi^2) = 0,825` ; depassement `+2,70` dB."""
    pulsation, surtension = so.resonance()
    assert pulsation == pytest.approx(math.sqrt(0.68), rel=1e-12)
    assert surtension == pytest.approx(2.70, abs=5e-3)


def test_ex6_a_la_pulsation_propre() -> None:
    """`|H(j)| = 10/0,8 = 12,5`, soit 21,94 dB, et l'argument vaut -90 degres."""
    gain, argument = so.point_propre()
    assert gain == pytest.approx(20 * math.log10(12.5), abs=1e-9)
    assert argument == pytest.approx(-90.0, abs=1e-9)


def test_ex6_le_sommet_est_bien_un_maximum() -> None:
    """Second chemin : le module exact, de part et d'autre de `w_r`."""
    pulsation, surtension = so.resonance()
    sommet = bo.gain_reel(so.TRANSMITTANCE, pulsation)
    assert sommet == pytest.approx(20.0 + surtension, abs=1e-9)
    for voisine in (0.95 * pulsation, 1.05 * pulsation):
        assert bo.gain_reel(so.TRANSMITTANCE, voisine) < sommet


def test_ex6_l_identification_vaut_pour_un_autre_second_ordre() -> None:
    """`18/(2 p^2 + 2 p + 8)` : `K = 9/4`, `w_n = 2`, `xi = 1/4`."""
    autre = 18 / (2 * P**2 + 2 * P + 8)
    assert so.parametres(autre) == (sp.Rational(9, 4), 2, sp.Rational(1, 4))
    pulsation, surtension = so.resonance(autre)
    assert pulsation == pytest.approx(math.sqrt(3.5), rel=1e-12)
    attendue = 20 * math.log10(1 / (2 * 0.25 * math.sqrt(1 - 0.0625)))
    assert surtension == pytest.approx(attendue, rel=1e-12)
