"""Planches de signaux et systemes : l'analyse harmonique.

Source : `SignauxSystemes/SLIDES/Slides_06_Analyse_harmonique.pdf`
(59 planches), lues sur les pages rendues.

Les oracles sont les points que les courbes affichent : pour
`10 / (5 j w + 1)`, 6,22 dB en 0,956 rad/s, -45 degres en 0,2 rad/s,
13,5 dB et -61,9 degres en 0,374 rad/s, le point (5, -5) de Nyquist ;
pour le second ordre `10 / (4 p^2 + 4 xi p + 1)`, les ordonnees -5, -10
et -16,7 en `w = 1 / T` ; les trois points du cercle du delai ; les
formules de `Re` et `Im` des deux exemples d'ordre quelconque.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.maths.domain.comparaison import (
    egales,
    en_degres,
    proches,
    proches_relativement,
)
from polytech.signaux.domain.bode import (
    TraceAsymptotique,
    classe_du_systeme,
    courbe_du_delai,
    courbe_inverse,
    depart_et_arrivee_de_nyquist,
    pulsation_de_coupure_du_second_ordre,
    pulsation_de_resonance,
    somme_des_courbes,
)
from polytech.signaux.domain.harmonique import (
    AnalyseHarmonique,
    courbe_elementaire_d_integrateur,
    courbe_elementaire_de_gain,
    courbe_elementaire_de_premier_ordre,
    courbe_elementaire_de_second_ordre,
    en_decibels,
    influence_du_gain,
    parties_reelle_et_imaginaire,
)
from polytech.signaux.domain.premier_ordre import PremierOrdre
from polytech.signaux.domain.reponse_indicielle import (
    bande_passante_du_second_ordre,
    pic_de_resonance_du_second_ordre,
)
from polytech.signaux.domain.second_ordre import SecondOrdre
from polytech.signaux.domain.stabilite import phase_ajoutee_par_un_retard
from polytech.signaux.domain.symboles import IMAGE, PULSATION
from polytech.signaux.domain.transmittance import Transmittance

P, W = IMAGE, PULSATION
PREMIER = AnalyseHarmonique(PremierOrdre(gain=10, constante=5).de_bode())


def _nombres(couple: tuple[sp.Expr, sp.Expr]) -> tuple[float, float]:
    return (float(couple[0]), float(couple[1]))


def test_les_points_de_bode_du_premier_ordre_de_la_planche_58() -> None:
    """« Frequency 0.956, Magnitude 6.22 » et « Frequency 0.2, Phase -45 »."""
    gain, _ = _nombres(PREMIER.point_de_bode(sp.Float(0.956)))
    assert proches(gain, 6.22, 0.01)
    gain, phase = _nombres(PREMIER.point_de_bode(sp.Rational(1, 5)))
    assert proches_relativement(phase, -45.0, 1e-6)
    assert proches(gain, 20 - 3.01, 0.01)


def test_le_point_de_black_du_premier_ordre_de_la_planche_58() -> None:
    """« Gain 13.5, Phase -61.9, Frequency 0.374 »."""
    phase, gain = _nombres(PREMIER.point_de_black(sp.Float(0.374)))
    assert proches(gain, 13.5, 0.05)
    assert proches(phase, -61.9, 0.05)


def test_le_demi_cercle_de_nyquist_du_premier_ordre() -> None:
    """Planche 63 : `Re = K / (1 + w^2 T^2)`, `Im = -K w T / (1 + w^2 T^2)`."""
    assert PREMIER.point_de_nyquist(sp.Rational(1, 5)) == (5, -5)
    assert PREMIER.point_de_nyquist(0) == (10, 0)
    for pulsation in (sp.Rational(1, 20), sp.Rational(2, 5), 1):
        reelle, imaginaire = PREMIER.point_de_nyquist(pulsation)
        assert (reelle - 5) ** 2 + imaginaire**2 == 25


def test_le_trace_asymptotique_du_premier_ordre() -> None:
    """Planche 60 : `20 log K`, puis -20 dB par decade a partir de `w = 1 / T`."""
    trace = TraceAsymptotique(gain=10, classe=0, cassures=((0.2, -1),))
    assert proches_relativement(trace.gain_en_decibels(0.02), 20.0, 1e-6)
    assert proches_relativement(trace.gain_en_decibels(2.0), 0.0, 1e-6)
    assert proches_relativement(trace.gain_en_decibels(20.0), -20.0, 1e-6)
    assert trace.pentes() == (0, -20)


@pytest.mark.parametrize(
    ("amortissement", "ordonnee"), [(1, -5), (0.5, -10), (0.3, -16.7)]
)
def test_le_nyquist_du_second_ordre_coupe_l_axe_imaginaire_en_un_sur_t(
    amortissement: float, ordonnee: float
) -> None:
    """Planche 70 : en `w = 1 / T`, `Re = 0` et `Im = -K / (2 xi)`."""
    systeme = SecondOrdre(10, 2, sp.nsimplify(amortissement))
    reelle, imaginaire = AnalyseHarmonique(systeme.de_bode()).point_de_nyquist(
        sp.Rational(1, 2)
    )
    assert reelle == 0
    assert proches(float(imaginaire), ordonnee, 0.05)


# Planche 68 : w_r = racine(1 - 2 xi^2) / T, avec xi = 0,3 et T = 2 s.
PULSATION_DE_RESONANCE_PLANCHE_68 = 0.45276925690687087


def test_la_resonance_du_second_ordre() -> None:
    """Planche 68 : `w_r = racine(1 - 2 xi^2) / T`, et le sommet de la courbe.

    La planche ecrit « |H(j w_r)| = 20 log(...) » : un module egale a des
    decibels. C'est le module qui vaut `K / (2 xi racine(1 - xi^2))`.
    """
    systeme = SecondOrdre(10, 2, sp.Rational(3, 10))
    resonance = pulsation_de_resonance(2.0, 0.3)
    assert proches_relativement(resonance, PULSATION_DE_RESONANCE_PLANCHE_68, 1e-6)
    analyse = AnalyseHarmonique(systeme.de_bode())
    sommet = float(analyse.module(sp.Float(resonance)))
    assert proches_relativement(
        sommet, 10 * pic_de_resonance_du_second_ordre(0.3), 1e-6
    )
    assert sommet > float(analyse.module(sp.Float(1.02 * resonance)))
    assert sommet > float(analyse.module(sp.Float(0.98 * resonance)))
    with pytest.raises(ValueError, match="pas de resonance"):
        pulsation_de_resonance(2.0, 0.8)


def test_la_pulsation_de_coupure_du_second_ordre() -> None:
    """Planche 68 : la pulsation ou le module vaut `K / racine(2)`."""
    coupure = pulsation_de_coupure_du_second_ordre(2.0, 0.3)
    systeme = SecondOrdre(10, 2, sp.Rational(3, 10))
    module = float(AnalyseHarmonique(systeme.de_bode()).module(sp.Float(coupure)))
    assert proches_relativement(module, 10 / math.sqrt(2), 1e-6)
    assert proches_relativement(coupure, bande_passante_du_second_ordre(0.5, 0.3), 1e-6)


def test_l_integrateur_et_le_derivateur_sont_inverses() -> None:
    """Planches 73 et 77 : 0 dB en `w = K`, ou en `w = 1 / K` ; -90 et +90 degres."""
    integrateur = influence_du_gain(courbe_elementaire_d_integrateur(), 10)
    assert integrateur[0].subs(W, 10) == 1
    assert integrateur[1] == -sp.pi / 2
    module, phase = courbe_inverse(courbe_elementaire_d_integrateur())
    assert (module, phase) == (W, sp.pi / 2)
    derivateur = influence_du_gain((module, phase), 10)
    assert derivateur[0].subs(W, sp.Rational(1, 10)) == 1


def test_les_zeros_ont_les_courbes_opposees_de_celles_des_poles() -> None:
    """Planches 92 a 95 : +20 et +40 dB par decade, +90 et +180 degres."""
    pole = courbe_elementaire_de_premier_ordre(sp.Integer(1))
    zero = courbe_inverse(pole)
    assert proches(float(en_decibels(zero[0]).subs(W, 100)), 40.0, 0.01)
    assert zero[1].subs(W, 1) == sp.pi / 4
    double = courbe_inverse(
        courbe_elementaire_de_second_ordre(sp.Integer(1), sp.Integer(1))
    )
    assert proches(float(en_decibels(double[0]).subs(W, 100)), 80.0, 0.01)
    assert double[1].subs(W, 1) == sp.pi / 2


def test_le_delai_ne_change_que_l_argument() -> None:
    """Planches 81 a 83 : module un, argument `-w tau` en radians, un cercle."""
    module, phase = courbe_du_delai(1)
    assert module == 1
    assert phase == phase_ajoutee_par_un_retard(1, W)
    cercle = {1: (0.54, -0.84), 2: (-0.42, -0.91), 3: (-0.99, -0.14)}
    for pulsation, (reelle, imaginaire) in cercle.items():
        angle = float(phase.subs(W, pulsation))
        assert proches(math.cos(angle), reelle, 0.005)
        assert proches(math.sin(angle), imaginaire, 0.005)
    assert proches(en_degres(float(phase.subs(W, 3))), -171.9, 0.05)


EXEMPLE = Transmittance(expression=10 * (1 + P) / (P * (1 + P / 5)))
"""Planches 96 a 102 : `10 (1 + p) / (p (1 + 0,2 p))`."""


def test_la_classe_d_un_systeme_compte_ses_integrateurs() -> None:
    """Planche 84 : `H = H_si / p^alpha`."""
    assert classe_du_systeme(EXEMPLE.expression) == 1
    assert classe_du_systeme(1 / (P + 1)) == 0
    assert classe_du_systeme(1 / (P**2 * (P + 1))) == 2


def test_le_trace_asymptotique_de_l_exemple_de_la_planche_96() -> None:
    """Un palier a 20 dB entre les cassures 1 et 5 rad/s."""
    trace = TraceAsymptotique(gain=10, classe=1, cassures=((1.0, 1), (5.0, -1)))
    assert trace.pentes() == (-20, 0, -20)
    assert proches_relativement(trace.gain_en_decibels(0.01), 60.0, 1e-6)
    assert proches_relativement(trace.gain_en_decibels(1.0), 20.0, 1e-6)
    assert proches_relativement(trace.gain_en_decibels(3.0), 20.0, 1e-6)
    assert proches(trace.gain_en_decibels(1000.0), -26.0, 0.05)
    reel = float(AnalyseHarmonique(EXEMPLE).point_de_bode(sp.Integer(1000))[0])
    assert proches(reel, trace.gain_en_decibels(1000.0), 0.01)
    with pytest.raises(ValueError, match="pulsation nulle ou negative"):
        trace.gain_en_decibels(0.0)


def test_les_courbes_de_bode_s_obtiennent_par_sommation() -> None:
    """Planches 87 et 88 : les decibels et les arguments s'ajoutent."""
    courbes = (
        courbe_elementaire_de_gain(sp.Integer(10)),
        courbe_elementaire_d_integrateur(),
        courbe_inverse(courbe_elementaire_de_premier_ordre(sp.Integer(1))),
        courbe_elementaire_de_premier_ordre(sp.Rational(1, 5)),
    )
    module, phase = somme_des_courbes(courbes)
    analyse = AnalyseHarmonique(EXEMPLE)
    for pulsation in (sp.Rational(1, 2), 3, 20):
        assert egales(module.subs(W, pulsation), analyse.module(pulsation))
        attendu = float(analyse.argument(pulsation))
        assert proches_relativement(float(phase.subs(W, pulsation)), attendu, 1e-6)
    with pytest.raises(ValueError, match="aucune courbe"):
        somme_des_courbes(())


def test_l_exemple_1_de_nyquist() -> None:
    """Planche 100 : `Re = -1 / (1 + w^2)`, `Im = -1 / (w (1 + w^2))`."""
    spectre = Transmittance(expression=1 / (P * (1 + P))).reponse_frequentielle()
    reelle, imaginaire = parties_reelle_et_imaginaire(spectre)
    assert egales(reelle, -(1 / (1 + W**2)))
    assert egales(imaginaire, -(1 / (W * (1 + W**2))))
    assert sp.limit(reelle, W, 0) == -1


def test_l_exemple_2_de_nyquist() -> None:
    """Planche 102 : `Re = 8 / (1 + 0,04 w^2)`, `Im = -10 (1 + 0,2 w^2) / (w (...))`."""
    reelle, imaginaire = parties_reelle_et_imaginaire(EXEMPLE.reponse_frequentielle())
    reduit = 1 + W**2 / 25
    assert egales(reelle, 8 / reduit)
    assert egales(imaginaire, -(10 * (1 + W**2 / 5) / (W * reduit)))
    assert sp.limit(reelle, W, 0) == 8


def test_le_depart_et_l_arrivee_d_une_courbe_de_nyquist() -> None:
    """Planche 98 : depart sur l'axe `-alpha 90`, arrivee sous `-(n - m) 90` degres."""
    assert depart_et_arrivee_de_nyquist(1 / (P * (1 + P))) == (-90, -180)
    assert depart_et_arrivee_de_nyquist(EXEMPLE.expression) == (-90, -90)
    assert depart_et_arrivee_de_nyquist(10 / (5 * P + 1)) == (0, -90)
    with pytest.raises(ValueError, match="degre du numerateur"):
        depart_et_arrivee_de_nyquist((P + 1) / (P + 2))


def test_un_gain_translate_la_courbe_d_amplitude() -> None:
    """Planches 103 a 106 : `20 log K` en plus, argument inchange."""
    base = AnalyseHarmonique(Transmittance(expression=1 / (P * (1 + P))))
    double = AnalyseHarmonique(Transmittance(expression=2 / (P * (1 + P))))
    for pulsation in (sp.Rational(1, 2), 3):
        gain, phase = _nombres(base.point_de_bode(pulsation))
        gain_double, phase_double = _nombres(double.point_de_bode(pulsation))
        assert proches_relativement(gain_double - gain, 20 * math.log10(2), 1e-6)
        assert proches_relativement(phase_double, phase, 1e-6)


def test_un_delai_ajoute_moins_omega_tau_a_l_argument() -> None:
    """Planches 107 a 110, `tau = 0,5 s` : attention aux unites, degres et radians."""
    sans = (1 / (W * sp.sqrt(1 + W**2)), -sp.pi / 2 - sp.atan(W))
    module, phase = somme_des_courbes((sans, courbe_du_delai(sp.Rational(1, 2))))
    assert module == sans[0]
    ecart = en_degres(float((phase - sans[1]).subs(W, 2)))
    assert proches(ecart, -57.3, 0.05)
