"""Les notes de cours de signaux et systemes, chapitre 1 : les signaux.

Source : `SignauxSystemes/SYLLABUS/Syllabus_Notes_de_cours_2016.pdf`,
pages 5 a 39 du fichier (chapitre 1, pages 1 a 35), lues sur les pages
rendues.

Les oracles : l'exemple 1 (`P = A^2 / 2`), le tableau 1.1 des decibels,
les exemples 3 et 6 (`SNR = 19.0679`, `12.8030 dB`), les 98 dB du disque
compact, la figure 1.16, l'exemple 9, les produits scalaires (1.41) a
(1.45) et les convolutions (1.49) a (1.52).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from polytech.signaux.domain.classification import (
    est_causal,
    partie_impaire,
    partie_paire,
    puissance_sur_une_periode,
    valeur_efficace_sur_une_periode,
)
from polytech.signaux.domain.echantillonnage import (
    echantillonner,
    frequence_apparente,
    frequence_minimale_d_echantillonnage,
    gain_en_decibels,
    puissance_estimee,
    quantifier,
)
from polytech.signaux.domain.elementaires import (
    rapport_signal_sur_bruit_d_un_quantificateur,
    rapport_signal_sur_bruit_en_decibels,
)
from polytech.signaux.domain.frequence import INSTANT, rect, tri
from polytech.signaux.domain.operations import (
    decaler,
    dilater,
    periodifier,
    produit_de_cosinus,
    reflechir,
    somme_de_sinusoides,
)
from polytech.signaux.domain.produits import (
    convolution_en,
    convolution_sur_la_droite,
    produit_scalaire_d_energie,
    produit_scalaire_periodique,
)

T = INSTANT
W0 = sp.Symbol("omega_0", positive=True)
PERIODE = 2 * sp.pi / W0


def test_un_signal_se_decompose_en_parties_paire_et_impaire() -> None:
    """Fig. 1.11 : `f = e + o`, `e(t) = (f(t) + f(-t)) / 2`."""
    signal = sp.exp(T) + T**3
    paire, impaire = partie_paire(signal), partie_impaire(signal)
    assert sp.simplify(paire - sp.cosh(T)) == 0
    assert sp.simplify(impaire - (sp.sinh(T) + T**3)) == 0
    assert sp.simplify(paire.subs(T, -T) - paire) == 0
    assert sp.simplify(impaire.subs(T, -T) + impaire) == 0
    assert sp.simplify(paire + impaire - signal) == 0


def test_un_signal_causal_est_nul_avant_zero() -> None:
    """Fig. 1.8."""
    assert est_causal(sp.Heaviside(T) * sp.exp(-T))
    assert not est_causal(sp.exp(-T))
    assert not est_causal(sp.Heaviside(-T))


def test_l_exemple_1_la_puissance_et_la_valeur_efficace_d_un_sinus() -> None:
    """(1.4) et (1.6) : `P = A^2 / 2`, `F_RMS = A / racine(2)`."""
    amplitude = sp.Symbol("A", positive=True)
    sinus = amplitude * sp.sin(W0 * T)
    assert puissance_sur_une_periode(sinus, PERIODE) == amplitude**2 / 2
    efficace = valeur_efficace_sur_une_periode(sinus, PERIODE)
    assert sp.simplify(efficace - amplitude / sp.sqrt(2)) == 0
    dephase = amplitude * sp.cos(W0 * T + sp.Rational(1, 3))
    assert (
        sp.simplify(puissance_sur_une_periode(dephase, PERIODE) - amplitude**2 / 2) == 0
    )
    with pytest.raises(ValueError, match="periode nulle ou negative"):
        puissance_sur_une_periode(sinus, 0)


def test_l_echantillonnage_mesure_le_signal_tous_les_t_e() -> None:
    """(1.7) : `x(n) = x(n T_e)`, et la condition de Shannon."""
    echantillons = echantillonner(sp.sin(2 * sp.pi * T), cadence=8.0, nombre=5)
    assert echantillons == pytest.approx(
        [0.0, math.sqrt(2) / 2, 1.0, math.sqrt(2) / 2, 0.0]
    )
    assert frequence_minimale_d_echantillonnage(20_000.0) == 40_000.0
    assert frequence_minimale_d_echantillonnage(20_000.0) < 44_100.0
    with pytest.raises(ValueError, match="cadence nulle ou negative"):
        echantillonner(sp.sin(T), cadence=0.0, nombre=5)


def test_la_figure_1_16_le_repliement_d_une_sinusoide_de_15_5_hz() -> None:
    """La legende dit « echantillonnage de 12 Hz », la figure montre 8,5 Hz.

    8,5 Hz est le miroir de 15,5 Hz par rapport a 12 Hz, que la figure
    nomme « Nyquist Frequency » : la cadence y est de 24 Hz. A 12 Hz, la
    frequence apparente serait 3,5 Hz.
    """
    assert frequence_apparente(15.5, 24.0) == pytest.approx(8.5)
    assert frequence_apparente(15.5, 12.0) == pytest.approx(3.5)
    assert frequence_apparente(4.0, 8.0) == pytest.approx(4.0)
    assert frequence_apparente(3.0, 8.0) == pytest.approx(3.0)


def test_le_repliement_se_voit_sur_les_echantillons() -> None:
    """Les deux sinusoides passent par les memes points."""
    haute = echantillonner(sp.cos(2 * sp.pi * 15.5 * T), cadence=24.0, nombre=12)
    basse = echantillonner(sp.cos(2 * sp.pi * 8.5 * T), cadence=24.0, nombre=12)
    assert haute == pytest.approx(basse)


def _sinus_de_l_exemple() -> np.ndarray:
    return 10 * np.sin(2 * np.pi * 440 * np.arange(1, 20001) / 10_000)


def test_l_exemple_3_la_quantification_sur_deux_bits() -> None:
    """`q = 2 max / 2^b = 5` ; l'erreur reste entre `-q / 2` et `q / 2`.

    Le quantificateur de l'exemple, `floor(x / q) q + q / 2`, rend cinq
    niveaux et non quatre : le sommet du sinus en occupe un a lui seul.
    """
    signal = _sinus_de_l_exemple()
    quantifie, pas = quantifier(signal, bits=2)
    assert pas == pytest.approx(5.0, abs=1e-3)
    assert np.max(np.abs(signal - quantifie)) <= pas / 2 + 1e-12
    assert len(np.unique(np.round(quantifie, 6))) == 5
    with pytest.raises(ValueError, match="nombre de bits nul ou negatif"):
        quantifier(signal, bits=0)


def test_l_exemple_6_le_rapport_signal_sur_bruit_de_la_quantification() -> None:
    """`SNR = 19.0679`, `SNR_dB = 12.8030`, contre 13,8 dB par (1.12)."""
    signal = _sinus_de_l_exemple()
    quantifie, _ = quantifier(signal, bits=2)
    rapport = float(np.var(signal, ddof=1) / np.var(signal - quantifie, ddof=1))
    assert rapport == pytest.approx(19.0679, abs=1e-4)
    assert gain_en_decibels(rapport) == pytest.approx(12.8030, abs=1e-4)
    assert rapport_signal_sur_bruit_d_un_quantificateur(2) == pytest.approx(
        13.8, abs=0.01
    )
    assert rapport_signal_sur_bruit_d_un_quantificateur(16) == pytest.approx(
        98.0, abs=0.1
    )


def test_la_puissance_s_estime_sur_les_echantillons() -> None:
    """(1.11) : `P = (1 / N) somme |f(n)|^2`, soit 50 pour un sinus de crete 10."""
    assert puissance_estimee(_sinus_de_l_exemple()) == pytest.approx(50.0, abs=1e-6)
    with pytest.raises(ValueError, match="aucun echantillon"):
        puissance_estimee(np.array([]))


@pytest.mark.parametrize(
    ("puissance", "amplitude", "decibels"),
    [
        (0.25, 0.5, -6),
        (0.5, 1 / math.sqrt(2), -3),
        (4, 2, 6),
        (100, 10, 20),
        (1e4, 100, 40),
    ],
)
def test_le_tableau_1_1_des_decibels(
    puissance: float, amplitude: float, decibels: float
) -> None:
    """(1.9) : `10 log(P_s / P_b) = 20 log(S_RMS / B_RMS)`."""
    assert gain_en_decibels(puissance) == pytest.approx(decibels, abs=0.03)
    assert gain_en_decibels(amplitude**2) == pytest.approx(decibels, abs=0.03)
    assert rapport_signal_sur_bruit_en_decibels(puissance, 1.0) == gain_en_decibels(
        puissance
    )
    with pytest.raises(ValueError, match="rapport nul ou negatif"):
        gain_en_decibels(0.0)


def test_l_exemple_9_la_lecture_d_une_grandeur_sinusoidale() -> None:
    """`U_max = 14,1 = 10 racine(2)` ; `omega_0 = 2 pi / 0,02` ; `theta = 1 rad`.

    Les notes ecrivent « 315 rad/s » pour 314,16, et « 1 rad/s » pour une
    phase, qui est en radians.
    """
    assert 10 * math.sqrt(2) == pytest.approx(14.1, abs=0.05)
    pulsation = 2 * math.pi / 0.02
    assert pulsation == pytest.approx(314.16, abs=0.01)
    assert pulsation * 0.0032 == pytest.approx(1.0, abs=0.01)
    avance = decaler(sp.cos(pulsation * T), -0.0032)
    assert float(avance.subs(T, 0)) == pytest.approx(math.cos(1.005), abs=1e-3)


def test_les_operations_elementaires_de_la_figure_1_34() -> None:
    """(1.32) a (1.34) : retard de 4 s, reflexion, dilatation d'un facteur 2."""
    signal = sp.Piecewise(
        (T + 4, (T >= -3) & (T < 0)), (4, (T >= 0) & (T <= 3)), (0, True)
    )
    assert decaler(signal, 4).subs(T, 5) == signal.subs(T, 1)
    assert reflechir(signal).subs(T, 2) == signal.subs(T, -2)
    assert dilater(signal, sp.Rational(1, 2)).subs(T, -4) == signal.subs(T, -2)
    assert dilater(signal, sp.Rational(1, 2)).subs(T, 6) == 4
    with pytest.raises(ValueError, match="facteur d'echelle nul"):
        dilater(signal, 0)


def test_l_exemple_10_la_somme_de_deux_sinusoides_de_meme_frequence() -> None:
    """(1.35) : l'amplitude et la phase sont celles de la somme des phaseurs."""
    amplitude, phase = somme_de_sinusoides((3, sp.pi / 6), (4, -sp.pi / 3))
    somme = 3 * sp.cos(W0 * T + sp.pi / 6) + 4 * sp.cos(W0 * T - sp.pi / 3)
    ecart = sp.lambdify(
        T, (somme - amplitude * sp.cos(W0 * T + phase)).subs(W0, 2), "numpy"
    )
    assert np.allclose(ecart(np.array([0.0, 0.7, 2.1])), 0, atol=1e-12)
    assert amplitude == 5


def test_l_exemple_13_le_produit_de_deux_sinusoides() -> None:
    """Fig. 1.40 : `cos(a) cos(b) = (cos(a + b) + cos(a - b)) / 2`."""
    haute, basse = produit_de_cosinus(10, 9)
    assert (haute, basse) == (19, 1)
    produit = sp.cos(10 * T) * sp.cos(9 * T)
    assert sp.simplify(produit - (sp.cos(haute * T) + sp.cos(basse * T)) / 2) == 0


def test_l_exemple_12_la_periodification() -> None:
    """(1.36) : `f_T0(t) = somme de f(t - k T_0)`."""
    periodique = periodifier(tri(T), 2, repliques=3)
    assert periodique.subs(T, 0) == 1
    assert periodique.subs(T, 2) == 1
    assert periodique.subs(T, 1) == 0
    assert periodique.subs(T, sp.Rational(5, 2)) == sp.Rational(1, 2)
    assert periodique.subs(T, -2) == 1
    assert periodique.subs(T, -4) == 1
    with pytest.raises(ValueError, match="periode nulle ou negative"):
        periodifier(tri(T), 0, repliques=3)


def test_les_produits_scalaires_evidents_de_la_figure_1_42() -> None:
    """(1.41) et (1.42)."""
    assert produit_scalaire_d_energie(rect(T), sp.Heaviside(T)) == sp.Rational(1, 2)
    retard = sp.Rational(1, 4)
    assert produit_scalaire_d_energie(sp.cos(T), sp.DiracDelta(T - retard)) == sp.cos(
        retard
    )
    assert produit_scalaire_d_energie(rect(T), rect(T)) == 1
    assert produit_scalaire_d_energie(rect(T), sp.I * rect(T)) == -sp.I


def test_le_sinus_et_le_cosinus_sont_orthogonaux() -> None:
    """(1.43) et (1.44) : 0, et 1/2 pour le cosinus avec lui-meme."""
    sinus, cosinus = sp.sin(W0 * T), sp.cos(W0 * T)
    assert produit_scalaire_periodique(sinus, cosinus, PERIODE) == 0
    assert produit_scalaire_periodique(cosinus, cosinus, PERIODE) == sp.Rational(1, 2)


def test_les_exponentielles_imaginaires_sont_orthonormees() -> None:
    """(1.45) : de norme un, et orthogonales d'un harmonique a l'autre."""
    fondamentale = sp.exp(sp.I * W0 * T)
    assert produit_scalaire_periodique(fondamentale, fondamentale, PERIODE) == 1
    for rang in (0, 2, 3):
        harmonique = sp.exp(sp.I * rang * W0 * T)
        assert produit_scalaire_periodique(fondamentale, harmonique, PERIODE) == 0


def test_l_exemple_1_15_la_convolution_de_deux_rectangles() -> None:
    """(1.49) : `rect * rect = tri`, instant par instant."""
    for instant in (-1, sp.Rational(-1, 2), 0, sp.Rational(1, 4), 1, 2):
        aire = convolution_en(rect(T), rect(T), instant)
        assert aire == tri(T).subs(T, instant)


def test_la_convolution_en_un_instant_est_celle_de_la_fonction() -> None:
    causale = sp.Heaviside(T) * sp.exp(-T)
    fonction = convolution_sur_la_droite(rect(T), causale)
    for instant in (sp.Rational(-1, 4), sp.Rational(1, 3), 2):
        ponctuelle = convolution_en(rect(T), causale, instant)
        assert sp.simplify(fonction.subs(T, instant) - ponctuelle) == 0


def test_la_convolution_reflechit_le_second_signal() -> None:
    """Deux signaux sans symetrie : une porte decalee et une exponentielle causale."""
    causale = sp.Heaviside(T) * sp.exp(-T)
    decalee = rect(T - 2)
    assert convolution_en(decalee, causale, 1) == 0
    attendue = sp.exp(-sp.Rational(1, 2)) - sp.exp(-sp.Rational(3, 2))
    assert sp.simplify(convolution_en(decalee, causale, 3) - attendue) == 0


def test_deux_impulsions_dans_une_convolution_sont_refusees() -> None:
    peigne = sp.DiracDelta(T) + sp.DiracDelta(T - 1)
    with pytest.raises(ValueError, match="une seule impulsion de Dirac attendue"):
        convolution_sur_la_droite(tri(T), peigne)


def test_la_convolution_est_commutative() -> None:
    """(1.47), sur une porte et une exponentielle causale."""
    causale = sp.Heaviside(T) * sp.exp(-T)
    directe = convolution_sur_la_droite(rect(T), causale)
    inverse = convolution_sur_la_droite(causale, rect(T))
    for instant in (sp.Rational(-1, 4), sp.Rational(1, 3), 2):
        assert sp.simplify(directe.subs(T, instant) - inverse.subs(T, instant)) == 0


def test_l_exemple_1_16_convoluer_par_un_dirac_deplace_le_signal() -> None:
    """(1.50) et (1.51) : `f(t) * delta(t - t_0) = f(t - t_0)`."""
    retard = sp.Rational(3, 2)
    deplace = convolution_sur_la_droite(tri(T), sp.DiracDelta(T - retard))
    for instant in (1, retard, 2, 3):
        assert (
            sp.simplify(deplace.subs(T, instant) - tri(T - retard).subs(T, instant))
            == 0
        )
