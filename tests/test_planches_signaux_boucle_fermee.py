"""Planches de signaux et systemes : la boucle fermee et sa stabilite.

Source : `SignauxSystemes/SLIDES/Slides_07_Boucle_fermee_Stabilite.pdf`
(59 planches), lues sur les pages rendues.

Les oracles : la cascade de reservoirs (0,37 puis 0,27 en boucle
ouverte, le PID `0,17 (1 + 399 p)^2 / (399 p (1 + 39,9 p))`), les
transmittances `5 / (50 p^3 + 65 p^2 + 16 p + 6)`, les racines que les
planches 137, 138 et 143 affichent, les croisements de Nyquist (0,124 et
0,566 rad/s, `-2,65 K` et `-0,252 K`), la limite `K < 3,96`, et les
quatre exemples supplementaires.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from polytech.signaux.domain.boucle_fermee import Boucle
from polytech.signaux.domain.harmonique import parties_reelle_et_imaginaire
from polytech.signaux.domain.marges import (
    Marges,
    marges_de_stabilite,
    passe_a_droite_du_point_critique,
)
from polytech.signaux.domain.poles_numeriques import est_stable
from polytech.signaux.domain.stabilite import (
    Stabilite,
    critere_simplifie_de_nyquist,
    equation_caracteristique,
    est_stable_par_les_racines,
    retard_critique,
)
from polytech.signaux.domain.symboles import GAIN, IMAGE, PULSATION
from polytech.signaux.domain.transmittance import Transmittance

P, W, K = IMAGE, PULSATION, GAIN
G1 = 5 / (1 + P)
G2 = 1 / ((1 + 5 * P) * (1 + 10 * P))
CUBIQUE = 50 * P**3 + 65 * P**2 + 16 * P


def _boucle(correcteur: sp.Expr) -> Boucle:
    return Boucle(correcteur=correcteur, actionneur=G1, procede=G2, capteur=1)


def test_les_transmittances_equivalentes_a_la_boucle_fermee() -> None:
    """Planche 128 : `T_R = C G_1 G_2 / (1 + G_s C G_1 G_2)`, `T_P = -G_2 / (...)`."""
    c, g1, g2, gs = sp.symbols("C G_1 G_2 G_s")
    boucle = Boucle(correcteur=c, actionneur=g1, procede=g2, capteur=gs)
    assert sp.simplify(boucle.ouverte() - gs * c * g1 * g2) == 0
    assert (
        sp.simplify(
            boucle.vis_a_vis_de_la_consigne() - c * g1 * g2 / (1 + gs * c * g1 * g2)
        )
        == 0
    )
    assert (
        sp.simplify(boucle.vis_a_vis_de_la_perturbation() + g2 / (1 + gs * c * g1 * g2))
        == 0
    )
    consigne, perturbation = sp.symbols("R P_e")
    attendue = boucle.vis_a_vis_de_la_consigne() * consigne
    attendue += boucle.vis_a_vis_de_la_perturbation() * perturbation
    assert sp.simplify(boucle.sortie(consigne, perturbation) - attendue) == 0


def test_l_exemple_de_la_planche_137_avec_un_correcteur_unite() -> None:
    """`T_R = 5 / (50 p^3 + 65 p^2 + 16 p + 6)`, `T_P = -(1 + p) / (...)`."""
    boucle = _boucle(1)
    assert sp.simplify(boucle.vis_a_vis_de_la_consigne() - 5 / (CUBIQUE + 6)) == 0
    assert (
        sp.simplify(boucle.vis_a_vis_de_la_perturbation() + (1 + P) / (CUBIQUE + 6))
        == 0
    )
    assert (
        sp.simplify(_boucle(10).vis_a_vis_de_la_consigne() - 50 / (CUBIQUE + 51)) == 0
    )


RACINES = {
    0.1: (-1.0135, -0.1433, 0.0953, True),
    1: (-1.1090, -0.0955, 0.3148, True),
    5: (-1.3486, 0.0243, 0.6205, False),
    10: (-1.5276, 0.1138, 0.8092, False),
}
"""Planche 143 : la racine reelle, puis la paire, et le verdict."""


@pytest.mark.parametrize("gain", sorted(RACINES))
def test_les_racines_de_l_equation_caracteristique(gain: float) -> None:
    """`50 p^3 + 65 p^2 + 16 p + 1 + 5 K = 0`."""
    reelle, partie_reelle, partie_imaginaire, stable = RACINES[gain]
    caracteristique = equation_caracteristique(Transmittance(_boucle(gain).ouverte()))
    assert sp.expand(caracteristique - (CUBIQUE + 1 + 5 * sp.nsimplify(gain))) == 0
    racines = np.roots([50, 65, 16, 1 + 5 * gain])
    seule = next(r for r in racines if abs(r.imag) < 1e-9)
    paire = next(r for r in racines if r.imag > 1e-9)
    assert seule.real == pytest.approx(reelle, abs=5e-5)
    assert paire.real == pytest.approx(partie_reelle, abs=5e-5)
    assert paire.imag == pytest.approx(partie_imaginaire, abs=5e-5)
    assert est_stable(racines) is stable
    assert _boucle(gain).est_stable() is stable


def test_l_exemple_2_est_stable_quel_que_soit_le_gain() -> None:
    """Planche 144 : `p^2 + p + 10 K = 0`, racines toujours a gauche."""
    boucle = Boucle(correcteur=K / P, actionneur=10, procede=1 / (1 + P), capteur=1)
    caracteristique = equation_caracteristique(Transmittance(boucle.ouverte()))
    assert sp.expand(caracteristique - (P**2 + P + 10 * K)) == 0
    for gain in (sp.Rational(1, 100), sp.Rational(1, 40), 3, 500):
        ouverte = Transmittance(boucle.ouverte().subs(K, gain))
        assert est_stable_par_les_racines(ouverte)


def test_les_parties_reelle_et_imaginaire_de_la_planche_149() -> None:
    """`Re = 5 K (1 - 65 w^2) / D`, `Im = 5 K w (50 w^2 - 16) / D`."""
    spectre = Transmittance(_boucle(K).ouverte()).reponse_frequentielle()
    reelle, imaginaire = parties_reelle_et_imaginaire(spectre)
    commun = (1 + W**2) * (1 + 25 * W**2) * (1 + 100 * W**2)
    assert sp.simplify(reelle - 5 * K * (1 - 65 * W**2) / commun) == 0
    assert sp.simplify(imaginaire - 5 * K * W * (50 * W**2 - 16) / commun) == 0
    coupe_l_axe_imaginaire = 1 / sp.sqrt(65)
    assert float(coupe_l_axe_imaginaire) == pytest.approx(0.124, abs=5e-4)
    ordonnee = imaginaire.subs(W, coupe_l_axe_imaginaire) / K
    assert float(ordonnee) == pytest.approx(-2.65, abs=0.005)


def test_la_limite_de_stabilite_par_nyquist_et_par_routh() -> None:
    """Planche 151 : `Im = 0` en 0,566 rad/s, `Re = -0,252 K`, stable si `K < 3,96`.

    L'abscisse vaut -0,2525 : la planche tronque, et « (-2,52 ; 0) » de meme.
    """
    stabilite = Stabilite(Transmittance(_boucle(K).ouverte()))
    assert float(stabilite.pulsation_du_croisement()) == pytest.approx(0.566, abs=5e-4)
    unitaire = float(stabilite.reel_au_croisement({K: 1}))
    assert unitaire == pytest.approx(-0.2525, abs=5e-5)
    assert math.trunc(unitaire * 1000) == -252
    decuple = float(stabilite.reel_au_croisement({K: 10}))
    assert decuple == pytest.approx(10 * unitaire)
    par_nyquist = stabilite.gain_critique_par_nyquist(K)
    assert float(par_nyquist) == pytest.approx(3.96, abs=0.005)
    assert sp.simplify(stabilite.gain_critique(K) - par_nyquist) == 0
    assert critere_simplifie_de_nyquist(unitaire)
    assert not critere_simplifie_de_nyquist(decuple)


EXEMPLES = {
    "a": (1 / (1 + 4 * P), True),
    "b": (1 / (4 * P**2 + P + 1), True),
    "c": (1 / (4 * P**2 + P), True),
    "d": (25 / (P * (P**2 + sp.Rational(3, 2) * P + 2)), False),
}
"""Planches 154 a 162 : la boucle ouverte, et le verdict en boucle fermee."""


@pytest.mark.parametrize("lettre", sorted(EXEMPLES))
def test_les_quatre_exemples_supplementaires(lettre: str) -> None:
    """Nyquist n'entoure pas (-1, 0) ; Black passe a droite de (-180 degres, 0 dB)."""
    ouverte, stable = EXEMPLES[lettre]
    assert est_stable_par_les_racines(Transmittance(ouverte)) is stable
    assert passe_a_droite_du_point_critique(ouverte) is stable


def test_les_marges_de_la_boucle_de_l_exemple_1() -> None:
    """Planches 166 et 169 : `MP = arg + 180` ou le module vaut un, `MG = 1 / |.|`."""
    marges = marges_de_stabilite(_boucle(1).ouverte())
    assert marges.pulsation_de_gain == pytest.approx(0.566, abs=5e-4)
    assert marges.de_gain == pytest.approx(3.96, abs=0.005)
    assert marges.de_gain_en_decibels() == pytest.approx(11.95, abs=0.02)
    module = abs(complex(_boucle(1).ouverte().subs(P, 1j * marges.pulsation_de_phase)))
    assert module == pytest.approx(1.0, abs=1e-6)
    assert 0 < marges.de_phase < 90
    assert isinstance(marges, Marges)


def test_les_marges_changent_de_signe_quand_la_boucle_devient_instable() -> None:
    """Remarques des planches 168 et 171 : `MP > 0` et `MG > 1` si stable en BF."""
    instable = marges_de_stabilite(_boucle(10).ouverte())
    assert instable.de_phase < 0
    assert instable.de_gain == pytest.approx(0.396, abs=5e-4)
    assert instable.de_gain_en_decibels() < 0
    limite = marges_de_stabilite(_boucle(sp.Float(3.96)).ouverte())
    assert limite.de_phase == pytest.approx(0.0, abs=0.05)


def test_un_delai_de_cinq_secondes_destabilise_la_boucle() -> None:
    """Planche 164 : stable pour `tau = 0`, instable pour `tau = 5`.

    La marge de phase donne le delai maximal admissible.
    """
    sans = marges_de_stabilite(_boucle(1).ouverte())
    admissible = float(
        retard_critique(np.radians(sans.de_phase), sans.pulsation_de_phase)
    )
    assert 0 < admissible < 5
    avec = marges_de_stabilite(_boucle(1).ouverte() * sp.exp(-5 * P))
    assert avec.de_phase < 0
    assert not passe_a_droite_du_point_critique(_boucle(1).ouverte() * sp.exp(-5 * P))
    juste = _boucle(1).ouverte() * sp.exp(-sp.Float(admissible) * P)
    assert marges_de_stabilite(juste).de_phase == pytest.approx(0.0, abs=0.05)


def test_deux_integrateurs_font_partir_l_argument_de_moins_180_degres() -> None:
    """Planche 98 : depart sur l'axe `-alpha 90` degres. Ici `1 / (p^2 (1 + p))`."""
    ouverte = 1 / (P**2 * (1 + P))
    marges = marges_de_stabilite(ouverte)
    pulsation = marges.pulsation_de_phase
    assert pulsation**2 * math.sqrt(1 + pulsation**2) == pytest.approx(1.0)
    assert marges.de_phase == pytest.approx(-math.degrees(math.atan(pulsation)))
    assert not est_stable_par_les_racines(Transmittance(ouverte))
    assert not passe_a_droite_du_point_critique(ouverte)


def test_une_boucle_sans_croisement_n_a_pas_de_marge_de_gain() -> None:
    """L'exemple a) n'atteint jamais -180 degres : marge de gain infinie."""
    marges = marges_de_stabilite(4 / (1 + 4 * P))
    assert marges.de_gain == float("inf")
    assert marges.de_phase == pytest.approx(180 - np.degrees(np.arctan(np.sqrt(15))))
    with pytest.raises(ValueError, match="module ne vaut jamais un"):
        marges_de_stabilite(1 / (2 + 4 * P))


RESERVOIRS = sp.Rational(9, 1000) * 826 / (1 + 304 * P) ** 3
"""Planche 122 : la vanne, le debit d'appoint, et trois reservoirs en cascade."""


def test_la_cascade_de_reservoirs_en_boucle_ouverte() -> None:
    """Planche 123 : 0,05 d'ouverture donne 0,37 ; `x_B = -0,1` ramene a 0,27."""
    gain = RESERVOIRS.subs(P, 0)
    assert float(gain * sp.Rational(5, 100)) == pytest.approx(0.37, abs=0.005)
    assert float(gain * sp.Rational(5, 100) - sp.Rational(1, 10)) == pytest.approx(
        0.27, abs=0.005
    )


def test_un_gain_seul_ne_donne_ni_precision_ni_stabilite_a_volonte() -> None:
    """Planche 124 : la boucle oscille des que le gain passe `8 / 7,434`."""
    stabilite = Stabilite(Transmittance(K * RESERVOIRS))
    critique = stabilite.gain_critique(K)
    assert float(critique) == pytest.approx(8 / 7.434, abs=1e-6)
    boucle = Boucle(
        correcteur=sp.Rational(1, 2), actionneur=1, procede=RESERVOIRS, capteur=1
    )
    finale = boucle.vis_a_vis_de_la_consigne().subs(P, 0)
    assert float(finale) == pytest.approx(0.788, abs=5e-4)


def test_le_pid_de_la_planche_126_stabilise_et_annule_l_erreur() -> None:
    """`C = 0,17 (1 + 399 p) / (399 p) (1 + 399 p) / (1 + 39,9 p)`."""
    pid = (
        sp.Rational(17, 100)
        * (1 + 399 * P) ** 2
        / (399 * P * (1 + sp.Rational(399, 10) * P))
    )
    boucle = Boucle(correcteur=pid, actionneur=1, procede=RESERVOIRS, capteur=1)
    assert boucle.est_stable()
    assert sp.limit(boucle.vis_a_vis_de_la_consigne(), P, 0) == 1
    assert sp.limit(boucle.vis_a_vis_de_la_perturbation(), P, 0) == 0
