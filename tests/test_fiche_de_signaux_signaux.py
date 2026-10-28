"""Fiche de signaux : les articles sur les signaux et les transformees.

Chaque formule de `cursus/fiches/SignauxSystemes.toml` est refaite par
`polytech.signaux` et comparee par `polytech.maths.domain.comparaison`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.comparaison import egales, proches
from polytech.maths.domain.laplace.theoremes import valeur_finale, valeur_initiale
from polytech.maths.domain.symboles import IMAGE as Q
from polytech.signaux.domain.classification import (
    puissance_sur_une_periode,
    valeur_efficace_sur_une_periode,
)
from polytech.signaux.domain.elementaires import rapport_signal_sur_bruit_en_decibels
from polytech.signaux.domain.forme_generale import mode_d_une_paire_conjuguee
from polytech.signaux.domain.frequence import INSTANT
from polytech.signaux.domain.harmonique import en_decibels
from polytech.signaux.domain.heaviside import original, residu
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import TEMPS as T

A = sp.Symbol("A", positive=True)
PERIODE = sp.Symbol("T_0", positive=True)


def test_puissance_d_un_cosinus_et_d_un_carre() -> None:
    """Article « Puissance et valeur efficace » : `A^2/2` et `A^2`."""
    cosinus = A * sp.cos(2 * sp.pi * INSTANT / PERIODE)
    carre = sp.Piecewise((A, INSTANT > 0), (-A, True))
    assert egales(puissance_sur_une_periode(cosinus, PERIODE), A**2 / 2)
    assert egales(puissance_sur_une_periode(carre, PERIODE), A**2)
    rms = valeur_efficace_sur_une_periode(cosinus, PERIODE)
    assert egales(rms, A / sp.sqrt(2))


def test_le_carre_et_le_sinus_sont_a_trois_decibels() -> None:
    """Cas de septembre 2016 : `A^2 / (A^2/2)` fait 3 dB."""
    ecart = rapport_signal_sur_bruit_en_decibels(2.0, 1.0)
    assert proches(ecart, 3.0, 0.011)


def test_la_periode_qui_donne_moins_vingt_decibels_est_200_secondes() -> None:
    """Cas de septembre 2019 : puissance `1/T_0` contre `1/2`."""
    periode = 200.0
    assert proches(rapport_signal_sur_bruit_en_decibels(1 / periode, 0.5), -20, 1e-9)


def test_la_valeur_efficace_d_un_motif_fixe_vaut_un_vingtieme_a_400_secondes() -> None:
    """Cas de juin 2022 : `1 / sqrt(T) = 1/20`."""
    periode = sp.Integer(400)
    motif = sp.Piecewise((1, sp.Abs(INSTANT) < sp.Rational(1, 2)), (0, True))
    assert egales(valeur_efficace_sur_une_periode(motif, periode), sp.Rational(1, 20))


def test_les_puissances_de_deux_harmoniques_s_ajoutent() -> None:
    """Cas de juin 2026 : `3 cos t - sin 2t` a pour puissance `9/2 + 1/2`."""
    signal = 3 * sp.cos(INSTANT) - sp.sin(2 * INSTANT)
    assert egales(puissance_sur_une_periode(signal, 2 * sp.pi), 5)


def test_le_decibel_compare_des_puissances_et_trois_decibels_doublent() -> None:
    """Article « Le rapport signal sur bruit »."""
    assert proches(rapport_signal_sur_bruit_en_decibels(2, 1), 3.0103, 1e-4)
    assert proches(float(en_decibels(sp.sqrt(2))), 3.0103, 1e-4)


def test_cent_vingt_huit_sources_egales_ajoutent_21_decibels() -> None:
    """Cas de juin 2018 : `10 log 128` ; un seul etudiant laisse 51 dB."""
    ajout = rapport_signal_sur_bruit_en_decibels(128, 1)
    assert proches(ajout, 21.07, 0.01)
    seul = 30 + ajout
    assert proches(seul, 51, 0.1)


def test_valeurs_initiale_et_finale_d_une_reponse_indicielle() -> None:
    """Article « valeurs initiale et finale » : `a(0+) = H(oo)`, `a(oo) = H(0)`."""
    transmittance = (2 * Q + 3) / (Q + 5)
    a_image = transmittance / Q
    assert egales(valeur_initiale(a_image), 2)
    assert egales(valeur_finale(a_image), sp.Rational(3, 5))
    temporel = original(a_image.subs(Q, P)).subs(T, 0)
    assert egales(temporel, valeur_initiale(a_image))


def test_le_developpement_d_heaviside_redonne_la_somme_des_exponentielles() -> None:
    """Article « Revenir au temps » : `A_i = N/D'`, `f = sum A_i e^{p_i t}`."""
    image = (3 * P + 5) / ((P + 1) * (P + 4))
    a1, a4 = residu(image, -1), residu(image, -4)
    assert egales(a1, sp.Rational(2, 3))
    assert egales(a4, sp.Rational(7, 3))
    attendu = a1 * sp.exp(-T) + a4 * sp.exp(-4 * T)
    assert egales(original(image), attendu)


def test_une_paire_conjuguee_donne_une_sinusoide_amortie() -> None:
    """Article « Lire les poles » : `2|A| e^{-sigma t} cos(omega t + arg A)`."""
    image = 1 / (P**2 + 2 * P + 5)
    pole = -1 + 2 * sp.I
    mode = mode_d_une_paire_conjuguee(residu(image, pole), pole)
    assert egales(mode, original(image))
