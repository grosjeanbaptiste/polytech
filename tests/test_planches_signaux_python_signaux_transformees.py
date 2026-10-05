"""Planches de signaux et systemes : les signaux et les transformees.

Sources : `SignauxSystemes/SLIDES/Slides_Python_01_Signaux.pdf` (42 pages)
et `Slides_Python_02_Transformees.pdf` (53 pages), lues sur les pages
rendues. Elles projettent les chapitres 1 et 2, deja portes par la
bibliotheque : ce fichier les y relie, et prend pour oracles ce qu'elles
ajoutent -- leurs extraits de code, executes tels quels
(`SNR_dB= 12.803033783286164`, les sommets 10, 30, 200 et 100 de
`freqz`), et leurs figures.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from polytech.maths.domain.fourier.transformee import PULSATION
from polytech.signaux.domain.classification import puissance_sur_une_periode
from polytech.signaux.domain.composantes import nature_de_la_composante
from polytech.signaux.domain.echantillonnage import (
    echantillonner,
    frequence_apparente,
    gain_en_decibels,
    plage_des_codes,
    puissance_estimee,
    quantifier,
)
from polytech.signaux.domain.elementaires import (
    rapport_signal_sur_bruit_d_un_quantificateur,
)
from polytech.signaux.domain.frequence import (
    FREQUENCE,
    INSTANT,
    en_frequence,
    rect,
    sinc,
    tri,
)
from polytech.signaux.domain.heaviside import residu
from polytech.signaux.domain.operations import decaler, periodifier
from polytech.signaux.domain.poles_numeriques import poles_et_zeros
from polytech.signaux.domain.produits import (
    convolution_en,
    convolution_sur_la_droite,
    produit_scalaire_d_energie,
    produit_scalaire_periodique,
)
from polytech.signaux.domain.raies import frequence_fondamentale
from polytech.signaux.domain.signaux_usuels import SIGNAUX_USUELS
from polytech.signaux.domain.spectre_estime import spectre_estime
from polytech.signaux.domain.symboles import IMAGE

T, F, P = INSTANT, FREQUENCE, IMAGE
W0 = sp.Symbol("omega_0", positive=True)
PERIODE = 2 * sp.pi / W0


def test_la_planche_10_integre_un_carre_de_sinus_mal_ecrit() -> None:
    """« (1 + sin(2 w t)) / 2 », etiquete « Erreur corrigee » : c'est `(1 - cos) / 2`.

    Les deux ont la meme moyenne, un demi, ce qui sauve le resultat ; mais
    seul le second est `sin^2`.
    """
    x = sp.Symbol("x", real=True)
    assert sp.simplify(sp.sin(x) ** 2 - (1 - sp.cos(2 * x)) / 2) == 0
    de_la_planche = (1 + sp.sin(2 * x)) / 2
    assert de_la_planche.subs(x, 0) == sp.Rational(1, 2)
    assert sp.sin(x).subs(x, 0) ** 2 == 0
    moyenne = sp.integrate(de_la_planche, (x, -sp.pi, sp.pi)) / (2 * sp.pi)
    assert moyenne == sp.Rational(1, 2)
    amplitude = sp.Symbol("A", positive=True)
    puissance = puissance_sur_une_periode(amplitude * sp.sin(W0 * T), PERIODE)
    assert puissance == amplitude**2 / 2


def test_l_exercice_du_signal_carre_sa_valeur_efficace_vaut_son_amplitude() -> None:
    """Planche 11 : « RMS = A »."""
    amplitude = sp.Symbol("A", positive=True)
    carre = sp.Piecewise((amplitude, T >= 0), (-amplitude, True))
    assert puissance_sur_une_periode(carre, PERIODE) == amplitude**2


def test_la_planche_13_une_sinusoide_a_la_moitie_de_la_cadence_disparait() -> None:
    """« Fe >= 2 Fmax » : a l'egalite, les echantillons du sinus sont tous nuls.

    C'est ce que montre la quatrieme ligne de la figure, 4 Hz a 8 Hz. La
    condition est stricte, comme les notes de cours l'ecrivent.
    """
    echantillons = echantillonner(sp.sin(2 * sp.pi * 4 * T), cadence=8.0, nombre=8)
    assert echantillons == pytest.approx([0.0] * 8, abs=1e-12)
    sous = echantillonner(sp.sin(2 * sp.pi * 3 * T), cadence=8.0, nombre=8)
    assert max(abs(valeur) for valeur in sous) == pytest.approx(1.0)


def test_la_planche_14_la_moitie_de_la_cadence_est_un_miroir() -> None:
    """24 Hz echantillonne a 24 Hz : des points alignes, une frequence nulle."""
    assert frequence_apparente(24.0, 24.0) == 0
    assert frequence_apparente(13.0, 24.0) == pytest.approx(11.0)
    echantillons = echantillonner(sp.cos(2 * sp.pi * 24 * T), cadence=24.0, nombre=6)
    assert echantillons == pytest.approx([1.0] * 6)


def test_le_disque_compact_code_sur_seize_bits() -> None:
    """Planche 17 : « 16 bits = [-32768, 32767] », soit `2^16` niveaux."""
    assert plage_des_codes(16) == (-32768, 32767)
    bas, haut = plage_des_codes(2)
    assert haut - bas + 1 == 2**2
    with pytest.raises(ValueError, match="nombre de bits nul ou negatif"):
        plage_des_codes(0)


def _sinus_du_code() -> np.ndarray:
    """`a=10; f=440; fe=10000; t=arange(0.0,1,1/fe)` : une seconde de signal."""
    instants = np.arange(0.0, 1, 1 / 10_000)
    return 10 * np.sin(2 * np.pi * 440 * instants)


def test_la_planche_22_la_puissance_d_une_seconde_de_sinus() -> None:
    """`signal_power=var(signal)` affiche 50, soit `a^2 / 2`."""
    signal = _sinus_du_code()
    assert float(np.var(signal)) == pytest.approx(50.0)
    assert puissance_estimee(signal) == pytest.approx(50.0)


def test_la_planche_28_le_rapport_signal_sur_bruit_affiche() -> None:
    """« SNR_dB= 12.803033783286164 », puis « RSB ~ 6 b » et 98 dB sur un CD."""
    signal = _sinus_du_code()
    quantifie, _ = quantifier(signal, bits=2)
    rapport = float(np.var(signal) / np.var(signal - quantifie))
    assert gain_en_decibels(rapport) == pytest.approx(12.803033783286164, abs=1e-9)
    assert gain_en_decibels(rapport) == pytest.approx(6 * 2, abs=1.0)
    assert rapport_signal_sur_bruit_d_un_quantificateur(16) == pytest.approx(
        98.0, abs=0.1
    )


def test_le_tableau_des_decibels_monte_jusqu_a_un_million() -> None:
    """Planche 24 : `10^6` en puissance, `10^3` en amplitude, 60 dB."""
    assert gain_en_decibels(1e6) == pytest.approx(60.0)
    assert gain_en_decibels(1000.0**2) == pytest.approx(60.0)


def test_la_puissance_d_une_exponentielle_imaginaire() -> None:
    """Planche 30 : `P = a^2`, sans le demi du sinus."""
    amplitude = sp.Symbol("a", positive=True)
    phaseur = amplitude * sp.exp(sp.I * W0 * T)
    assert puissance_sur_une_periode(phaseur, PERIODE) == amplitude**2


def test_les_produits_scalaires_poses_en_question() -> None:
    """Planches 40, 41 et 44 : `a / 2`, `sinc(tau)`, puis un demi."""
    largeur = sp.Symbol("a", positive=True)
    assert produit_scalaire_d_energie(rect(T / largeur), sp.Heaviside(T)) == largeur / 2
    instant = sp.Rational(1, 3)
    mesure = produit_scalaire_d_energie(sinc(T), sp.DiracDelta(T - instant))
    assert sp.simplify(mesure - sinc(instant)) == 0
    phaseur, cosinus = sp.exp(sp.I * W0 * T), sp.cos(W0 * T)
    assert produit_scalaire_periodique(phaseur, cosinus, PERIODE) == sp.Rational(1, 2)


def test_les_convolutions_de_la_planche_47() -> None:
    """Deux portes de hauteur 2 : un triangle qui culmine a 4 en `t = 1`."""
    porte = 2 * rect(T - sp.Rational(1, 2))
    assert convolution_en(porte, porte, 1) == 4
    assert convolution_en(porte, porte, sp.Rational(1, 2)) == 2
    assert convolution_en(porte, porte, 2) == 0
    decroissante = sp.Heaviside(T) * sp.exp(-T)
    sommet = convolution_en(porte, decroissante, 1)
    assert sp.simplify(sommet - 2 * (1 - sp.exp(-1))) == 0
    assert float(sommet) == pytest.approx(1.26, abs=0.005)


def test_convoluer_par_un_peigne_periodifie() -> None:
    """Planche 49 : `f * delta(t - t_0) = f(t - t_0)`, et la somme des repliques."""
    repliques = [
        convolution_sur_la_droite(tri(T), sp.DiracDelta(T - 2 * rang))
        for rang in (-1, 0, 1)
    ]
    for instant in (1, sp.Rational(3, 2), 2, 3):
        attendu = decaler(tri(T), 2).subs(T, instant)
        assert repliques[2].subs(T, instant) == attendu
    somme = sum(repliques, sp.S.Zero)
    periodique = periodifier(tri(T), 2, repliques=1)
    for instant in (sp.Rational(-3, 2), 0, sp.Rational(1, 2), 2):
        assert sp.simplify((somme - periodique).subs(T, instant)) == 0


def test_la_planche_18_la_porte_en_notation_omega() -> None:
    """`rect(t) <-> sin(w / 2) / (w / 2) = sinc(f)`."""
    en_pulsation = sp.sin(PULSATION / 2) / (PULSATION / 2)
    ecart = sp.lambdify(F, en_frequence(en_pulsation, PULSATION) - sinc(F), "numpy")
    assert np.allclose(ecart(np.array([-1.3, 0.4, 2.6])), 0, atol=1e-12)


def _sommet(echantillons: np.ndarray, frequence: float = 0.0) -> float:
    """`abs(freqz(signal))` a une frequence, pour une cadence de 10 Hz."""
    return float(spectre_estime(echantillons, 10.0, np.array([frequence]))[0] * 10)


def test_les_sommets_des_estimations_des_planches_29_a_32() -> None:
    """`rect` et `tri` a 10 Hz : un sommet de 10, la surface fois la cadence."""
    instants = np.arange(-1, 1, 1 / 10)
    porte = np.where(np.abs(instants) <= 0.5, 1.0, 0.0)
    assert _sommet(porte) == pytest.approx(10.0)
    larges = np.arange(-2, 2, 1 / 10)
    triangle = np.where(np.abs(larges) <= 1, 1 - np.abs(larges), 0.0)
    assert _sommet(triangle) == pytest.approx(10.0)


def test_trois_periodes_font_un_spectre_de_raies() -> None:
    """Planches 44 et 45 : sommet 30 ; la raie de 0,5 Hz monte vers `N F_1`."""
    instants = np.arange(-1, 1, 1 / 10)
    porte = np.where(np.abs(instants) <= 0.5, 1.0, 0.0)
    carre = np.concatenate((porte, porte, porte))
    assert _sommet(carre) == pytest.approx(30.0)
    assert _sommet(carre, 0.5) == pytest.approx(len(carre) / math.pi, rel=0.01)
    triangle = np.where(np.abs(instants) <= 1, 1 - np.abs(instants), 0.0)
    assert _sommet(np.concatenate((triangle, triangle, triangle))) == pytest.approx(
        30.0
    )


def test_le_signal_constant_a_un_sommet_de_200() -> None:
    """Planche 41 : deux secondes a 100 Hz, `exp(1j*2*pi*0*t)`."""
    constant = np.ones(200)
    assert spectre_estime(constant, 100.0, np.array([0.0]))[0] * 100 == pytest.approx(
        200.0
    )


def test_le_segment_de_parole_de_la_planche_46() -> None:
    """`range(600, 839)` a 8 kHz : 30 ms, soit pres de quatre periodes de 65."""
    echantillons = len(range(600, 839))
    assert echantillons / 8000 == pytest.approx(0.030, abs=2e-4)
    periodes = echantillons / 65
    assert periodes == pytest.approx(3.7, abs=0.05)
    assert frequence_fondamentale(65, 8000.0) == pytest.approx(123.0, abs=0.1)


def test_les_poles_et_le_zero_de_la_planche_52() -> None:
    """`F(p) = (p + 1) / (p^2 + p + 0,5)` : poles `-0,5 +/- 0,5 j`, zero -1."""
    poles, zeros, _ = poles_et_zeros([1.0, 1.0], [1.0, 1.0, 0.5])
    assert sorted(poles.imag) == pytest.approx([-0.5, 0.5])
    assert poles.real == pytest.approx([-0.5, -0.5])
    assert zeros == pytest.approx([-1.0])
    assert nature_de_la_composante(complex(poles[0])) == "sinusoide amortie"


def test_le_tableau_des_planches_53_a_57_est_celui_du_syllabus() -> None:
    assert len(SIGNAUX_USUELS) == 13
    assert (
        SIGNAUX_USUELS["exponentielle fois t"].image
        == 1 / (P + sp.Symbol("a", positive=True)) ** 2
    )


def test_la_planche_61_deux_poles_conjugues_font_un_cosinus_et_un_sinus() -> None:
    """`2 Re(A exp(j 2 t)) = 2 (Re(A) cos(2 t) - Im(A) sin(2 t))`."""
    amplitude = residu((P + 1) / (P**2 + 4), 2 * sp.I)
    assert (2 * sp.re(amplitude), -2 * sp.im(amplitude)) == (1, sp.Rational(1, 2))
    gauche = 2 * sp.re(amplitude * sp.exp(2 * sp.I * T))
    assert sp.simplify(gauche - (sp.cos(2 * T) + sp.sin(2 * T) / 2)) == 0


def test_la_carte_des_poles_de_la_planche_65() -> None:
    """« x : pole simple, X : pole multiple »."""
    assert nature_de_la_composante(-2) == "exponentielle decroissante"
    assert nature_de_la_composante(-1 + 3j) == "sinusoide amortie"
    assert (
        nature_de_la_composante(3j, double=True)
        == "sinusoide non amortie, multipliee par t"
    )
    assert nature_de_la_composante(0, double=True) == "terme constant, multiplie par t"
    assert nature_de_la_composante(1 + 3j) == "sinusoide amplifiee"
    assert nature_de_la_composante(2) == "exponentielle croissante"
