"""Signaux et systemes : le recueil d'exercices, poles et transmittances.

Source : `SignauxSystemes/EXS/Exs_Recueil.pdf`. Ce fichier couvre les
planches 4, 5, 6, 9, 11, 12 et 14 : des poles, on tire l'allure d'une
reponse, et inversement. Les reponses imprimees sur les planches 11, 12 et
14 sont recopiees comme oracles.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.signaux.application import recueil_graphes as gr
from polytech.signaux.application import recueil_poles as po
from polytech.signaux.domain.symboles import IMAGE as P

# --- Planche 4 : dessiner une sinusoide amortie -----------------------------


def test_p4_les_reperes_du_dessin() -> None:
    """`sigma = 100`, `w = 2 pi 100` : la constante de temps vaut la periode."""
    periode, constante, depart = po.reperes_de_la_sinusoide()
    assert periode == pytest.approx(0.01)
    assert constante == pytest.approx(0.01)
    assert depart == pytest.approx(math.sqrt(2) / 2)


def test_p4_l_amplitude_perd_63_pour_cent_par_periode() -> None:
    """Une constante de temps par periode : l'enveloppe tombe a `1/e`."""
    assert po.enveloppe_apres(periodes=1) == pytest.approx(math.exp(-1))
    assert po.enveloppe_apres(periodes=5) < 0.01


# --- Planche 5 : l'allure sans inverser Laplace -----------------------------


def test_p5_les_deux_theoremes() -> None:
    """`u(0+) = 1` et `u(infini) = 0`."""
    assert po.valeurs_extremes() == (1, 0)


def test_p5_les_poles() -> None:
    """`-1 +- j` : une constante de temps d'une seconde, une periode de `2 pi`."""
    assert po.sigma_et_omega(P**2 + 2 * P + 2) == (1, 1)


def test_p5_par_l_inversion_formelle() -> None:
    """Second chemin : `u(t) = e^(-t) (cos t - sin t)`, qui s'annule en `pi/4`."""
    instant = sp.Symbol("t", nonnegative=True)
    attendue = sp.exp(-instant) * (sp.cos(instant) - sp.sin(instant))
    assert sp.simplify(po.original_de_la_tension(instant) - attendue) == 0
    assert po.original_de_la_tension(sp.pi / 4) == 0


# --- Planche 6 : apparier transmittances, reponses et diagrammes ------------


def test_p6_l_appariement() -> None:
    """`A-e-iii`, `B-b-v`, `C-a-ii`, `D-d-iv`, `E-c-i`."""
    assert po.appariement_des_transmittances() == {
        "A": ("e", 3),
        "B": ("b", 5),
        "C": ("a", 2),
        "D": ("d", 4),
        "E": ("c", 1),
    }


# --- Planche 9 : une reponse impulsionnelle sans oscillation ----------------


def test_p9_la_valeur_limite_de_alpha() -> None:
    """Poles reels des que `alpha^2 >= 800` : `alpha >= 20 racine(2)`."""
    assert gr.alpha_limite() == 20 * sp.sqrt(2)
    assert float(gr.alpha_limite()) == pytest.approx(28.28, abs=5e-3)


def test_p9_de_part_et_d_autre_de_la_limite() -> None:
    """A 28, les poles sont complexes ; a 29, reels."""
    assert not all(pole.is_real for pole in gr.poles_pour_alpha(28))
    assert all(pole.is_real for pole in gr.poles_pour_alpha(29))


# --- Planche 11 : lire des poles sur une reponse impulsionnelle -------------


def test_p11_la_reponse_de_la_planche() -> None:
    """Planche 11 : « -1 +- j 2 pi », a la precision d'une lecture."""
    sigma, omega = po.poles_lus_sur_la_reponse()
    assert omega == pytest.approx(2 * math.pi)
    assert sigma == pytest.approx(1, abs=0.1)


def test_p11_la_lecture_se_verifie() -> None:
    """Second chemin : `e^(-t)` ramene la premiere crete a la deuxieme."""
    (_, premiere), (_, deuxieme), _ = po.CRETES
    assert premiere * math.exp(-1) == pytest.approx(deuxieme, abs=0.05)


# --- Planche 12 : deux cellules C-R en cascade ------------------------------


def test_p12_la_transmittance_de_la_planche() -> None:
    """Planche 12 : `p^2 / (p^2 + 3p/RC + 1/R^2C^2)`."""
    tau = po.R * po.C
    attendue = P**2 / (P**2 + 3 * P / tau + 1 / tau**2)
    assert sp.simplify(po.transmittance_des_deux_cellules() - attendue) == 0


def test_p12_les_limites_de_la_planche() -> None:
    """Planche 12 : « infini, 0 » pour `h(0)` et `h(infini)`."""
    assert po.limites_de_l_impulsionnelle() == (sp.oo, 0)


def test_p12_deux_poles_reels() -> None:
    """Planche 12 : deux poles reels ; `(-3 +- racine 5)/(2RC)`, RC = 10 ms."""
    lent, rapide = po.poles_des_deux_cellules()
    assert lent == pytest.approx(-38.2, abs=5e-2)
    assert rapide == pytest.approx(-261.8, abs=5e-2)


def test_p12_l_extinction() -> None:
    """Planche 12 : apres cinq fois la plus grande constante de temps."""
    constante, duree = po.extinction()
    assert constante == pytest.approx(26.2e-3, abs=5e-5)
    assert duree == pytest.approx(0.131, abs=5e-4)


# --- Planche 14 : un graphe a quatre integrateurs ---------------------------


def test_p14_la_transmittance_de_la_planche() -> None:
    """Planche 14 : `(k p^3 + p^2 + p + 1)/(p^4 + 10 p^3 + 100 p^2 + 200 p - m)`."""
    haut = gr.K * P**3 + P**2 + P + 1
    bas = P**4 + 10 * P**3 + 100 * P**2 + 200 * P - gr.M
    assert sp.simplify(gr.transmittance_a_quatre_integrateurs() - haut / bas) == 0


def test_p14_les_deux_reglages_de_la_planche() -> None:
    """Planche 14 : `k = 1` et `m = -1/2`."""
    assert gr.reglages() == {gr.K: 1, gr.M: sp.Rational(-1, 2)}


def test_p14_la_valeur_finale_est_moins_un_sur_m() -> None:
    """La planche ecrit « 1/m » ; c'est `-1/m`, et `m = -1/2` donne bien 2."""
    assert gr.valeur_finale_de_l_indicielle() == -1 / gr.M
    assert gr.valeur_finale_de_l_indicielle().subs(gr.reglages()) == 2


def test_p14_le_systeme_regle_est_stable() -> None:
    """Sans quoi le theoreme de la valeur finale ne s'appliquerait pas."""
    assert gr.est_stable_une_fois_regle()
