"""Signaux et systemes : les exercices resolus, chapitre 2 (0.11 a 0.21).

Source : `SignauxSystemes/EXS/Exs_01_02_04_05_Resolus.pdf`, pages 5 a 8.
Les solutions imprimees sont recopiees comme oracles ; chacune est
recoupee par un autre chemin -- transformer la reponse imprimee et
retrouver l'image, inverser le spectre et retrouver le signal. Toutes
sont justes ; l'exercice 0.12, « a trouver vous-meme », recoit sa reponse.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.signaux.application import resolus_fourier as fo
from polytech.signaux.application import resolus_inverses as iv
from polytech.signaux.application import resolus_laplace as la
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import TEMPS as T

# --- 0.12 a 0.14 : spectres de raies, modulation ----------------------------


def test_0_12_les_raies_de_la_porte_repetee() -> None:
    """Sans solution imprimee : `X_k = (1/5) sinc(k/5)`, une raie par hertz."""
    assert fo.raie_de_la_porte(0) == sp.Rational(1, 5)
    assert fo.raie_de_la_porte(5) == 0
    assert float(fo.raie_de_la_porte(1)) == pytest.approx(
        math.sin(math.pi / 5) / math.pi
    )


def test_0_12_la_raie_centrale_est_la_valeur_moyenne() -> None:
    """Second chemin : une porte d'un cinquieme de seconde par seconde."""
    assert fo.raie_de_la_porte(0) == fo.LARGEUR_DE_LA_PORTE / fo.PERIODE_DE_LA_PORTE


def test_0_13_les_vingt_six_decibels_de_la_solution() -> None:
    """Page 7 : la raie centrale est marquee -26 dB, les raies tous les 0,1 Hz."""
    assert fo.niveau_de_la_raie_centrale() == pytest.approx(-26, abs=0.05)
    assert fo.ecart_des_raies() == sp.Rational(1, 10)


def test_0_14_le_spectre_de_la_solution() -> None:
    """Page 7 : deux portes de hauteur 1/2, de `f0 - 1/2` a `f0 + 1/2`."""
    assert fo.bords_du_spectre_module() == (fo.F0 - fo.DEMI, fo.F0 + fo.DEMI)
    assert fo.spectre_module(fo.PORTEUSE_D_ESSAI) == sp.Rational(1, 2)
    assert fo.spectre_module(fo.PORTEUSE_D_ESSAI + 1) == 0
    assert fo.spectre_module(0) == 0


def test_0_14_une_porteuse_trop_basse_fait_se_recouvrir_les_lobes() -> None:
    """La figure suppose `f0 > 1/2` ; a un quart de hertz, les moities s'ajoutent."""
    assert not fo.les_lobes_se_recouvrent(fo.PORTEUSE_D_ESSAI)
    assert fo.les_lobes_se_recouvrent(sp.Rational(1, 4))
    assert fo.spectre_module(0, sp.Rational(1, 4)) == 1


def test_0_14_le_spectre_redonne_le_signal_module() -> None:
    """Second chemin : inverser le spectre rend `sinc(t) cos(2 pi f0 t)`."""
    for instant in (0.3, 1.7):
        attendu = math.sin(math.pi * instant) / (math.pi * instant)
        attendu *= math.cos(2 * math.pi * fo.PORTEUSE_D_ESSAI * instant)
        assert fo.signal_depuis_le_spectre_module(instant) == pytest.approx(attendu)


# --- 0.15 et 0.16 : deux transformees de Laplace lues sur un graphique ------


def test_0_15_l_image_de_la_solution() -> None:
    """Page 8 : `F(p) = (4p - 1,25) / (p (p + 1,25))`."""
    attendue = (4 * P - sp.Rational(5, 4)) / (P * (P + sp.Rational(5, 4)))
    assert sp.simplify(la.image_15() - attendue) == 0


def test_0_15_les_deux_theoremes() -> None:
    """La valeur initiale, 4, et la valeur finale, -1, se lisent sur la figure."""
    assert la.limites(la.image_15()) == (4, -1)
    assert (la.signal_15().subs(T, 0), sp.limit(la.signal_15(), T, sp.oo)) == (4, -1)


def test_0_15_la_tangente_a_l_origine() -> None:
    """Page 6 : elle coupe l'asymptote en `t = 0,8`, la constante de temps."""
    assert la.instant_ou_la_tangente_coupe_l_asymptote() == sp.Rational(4, 5)


def test_0_16_l_image_de_la_solution() -> None:
    """Page 8 : `2/p - 1/(p+1) + e^(-8p) (2/(p + 1/2) - 2/p)`."""
    retard = sp.exp(-8 * P) * (2 / (P + sp.Rational(1, 2)) - 2 / P)
    attendue = 2 / P - 1 / (P + 1) + retard
    assert sp.simplify(la.image_16() - attendue) == 0


def test_0_16_les_reperes_de_la_figure() -> None:
    """Page 6 : part de 1, tend vers 2, puis decroit depuis 2 a partir de 8."""
    assert la.signal_16(0) == 1
    assert float(la.signal_16(8)) == pytest.approx(2, abs=1e-3)
    assert float(la.signal_16(10)) == pytest.approx(2 * math.exp(-1), abs=1e-3)
    assert la.limites(la.image_16())[0] == 1


def test_0_16_ce_que_l_image_neglige() -> None:
    """A `t = 8` la montee n'a pas tout a fait atteint 2 : il manque `e^-8`."""
    assert la.ecart_au_raccord() == sp.exp(-8)
    assert float(la.ecart_au_raccord()) < 1e-3


# --- 0.17 a 0.21 : des inverses de Laplace ----------------------------------

E = sp.exp


@pytest.mark.parametrize(
    ("nom", "solution"),
    [
        ("0.17a", E(-T) - E(-2 * T)),
        ("0.17b", sp.sin(2 * T) / 2),
        ("0.17c", -sp.Rational(1, 100) + T / 10 + E(-10 * T) / 100),
        ("0.19a", sp.DiracDelta(T) + 2 * E(-2 * T) - 8 * E(-4 * T)),
        (
            "0.19b",
            (sp.Rational(9, 49) - sp.Rational(2, 7) * T) * E(-3 * T)
            - sp.Rational(9, 49) * E(-10 * T),
        ),
        ("0.21", (E(2 * T) - E(-2 * T)) / 32 + sp.cos(2 * T - 3 * sp.pi / 2) / 16),
    ],
)
def test_les_originaux_des_solutions(nom: str, solution: sp.Expr) -> None:
    """Page 8 : l'original imprime, retrouve par le developpement d'Heaviside."""
    assert sp.simplify(iv.original_de(nom) - solution) == 0


@pytest.mark.parametrize("nom", ["0.17a", "0.17b", "0.17c", "0.19b", "0.21"])
def test_les_originaux_se_transforment_en_leur_image(nom: str) -> None:
    """Second chemin : la transformee de l'original rend l'image de l'enonce."""
    assert sp.simplify(iv.image_de_l_original(nom) - iv.IMAGES[nom]) == 0


def test_0_18_l_original_retarde_de_la_solution() -> None:
    """Page 8 : `0,1 (t-5) + 0,09 - 0,09 e^(-10 (t-5))`, a partir de `t = 5`."""
    retarde = T - 5
    attendu = retarde / 10 + sp.Rational(9, 100) * (1 - E(-10 * retarde))
    assert sp.simplify(iv.original_retarde() - attendu) == 0
    assert iv.RETARD == 5


def test_0_20_l_original_de_la_solution() -> None:
    """Page 8 : `delta(t) - 2,3 e^(-3t) + 0,316 cos(t - 0,322)`."""
    exponentielle, amplitude, phase = iv.forme_de_la_solution_20()
    assert exponentielle == sp.Rational(-23, 10)
    assert float(amplitude) == pytest.approx(0.316, abs=5e-4)
    assert float(phase) == pytest.approx(0.322, abs=5e-4)
    assert iv.original_de("0.20").has(sp.DiracDelta)


def test_0_21_le_theoreme_de_la_valeur_finale_ne_s_applique_pas() -> None:
    """Page 8 : un pole a droite de l'axe imaginaire, donc `f` n'est pas bornee."""
    assert iv.poles_a_droite("0.21") == (2,)
    assert sp.limit(P * iv.IMAGES["0.21"], P, 0) == 0
    assert sp.limit(iv.original_de("0.21"), T, sp.oo) == sp.oo
