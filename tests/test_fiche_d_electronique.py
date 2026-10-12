"""La fiche de revision d'electronique : ses formules refaites par un autre chemin.

Les reponses des examens sont deja refaites dans
`tests/test_examen_electronique_*.py`. Ce fichier tient ce que les articles
de `cursus/fiches/ElectroniqueInstrumentation.toml` affirment en plus : les
resultats que la fiche derive elle-meme, chacun recalcule ici par la
bibliotheque ou par sympy, sans recopier la formule de l'article.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electronique.domain import abcd
from polytech.electronique.domain.amplification import (
    gain_maximal_par_etage,
    nombre_d_etages,
)
from polytech.electronique.domain.echelle_lc import elements_de_butterworth
from polytech.electronique.domain.friis import facteur_en_cascade
from polytech.electronique.domain.quantification import Quantificateur
from polytech.electronique.domain.transpositions import mise_a_l_echelle_d_impedance

s, t = sp.symbols("s t", positive=True)
R, C, G, Q, w0 = sp.symbols("R C G Q omega_0", positive=True)


def test_deux_cellules_rc_en_cascade_ne_donnent_pas_le_carre_d_une_seule() -> None:
    """Article « La matrice ABCD » : `A = s^2 R^2 C^2 + 3 s R C + 1`."""
    cellule = abcd.cascade(abcd.serie(R), abcd.derivation(1 / (s * C)))
    a_vide = sp.simplify(1 / abcd.cascade(cellule, cellule)[0, 0])
    assert sp.simplify(1 / a_vide - (s**2 * R**2 * C**2 + 3 * s * R * C + 1)) == 0
    assert sp.simplify(a_vide - 1 / (1 + s * R * C) ** 2) != 0


def test_l_admittance_d_entree_a_sortie_ouverte() -> None:
    """Article « Les parametres z et y » : `y11 - y12 y21 / y22`."""
    y11, y12, y21, y22, v1, v2 = sp.symbols("y11 y12 y21 y22 V1 V2")
    sortie_ouverte = sp.solve(sp.Eq(y21 * v1 + y22 * v2, 0), v2)[0]
    entree = (y11 * v1 + y12 * sortie_ouverte) / v1
    assert sp.simplify(entree - (y11 - y12 * y21 / y22)) == 0


def _second_ordre(variable: sp.Expr) -> sp.Expr:
    return G * w0**2 / (variable**2 + (w0 / Q) * variable + w0**2)


def test_le_facteur_de_qualite_est_le_gain_en_omega_zero_sur_le_gain_continu() -> None:
    assert sp.simplify(sp.Abs(_second_ordre(sp.I * w0)) - G * Q) == 0
    assert _second_ordre(0) == G


def test_omega_zero_n_est_la_coupure_que_pour_q_egal_a_un_sur_racine_de_deux() -> None:
    assert sp.solve(sp.Eq(G * Q, G / sp.sqrt(2)), Q) == [sp.sqrt(2) / 2]


@pytest.mark.parametrize(("qualite", "reels"), [(0.4, True), (0.5, True), (0.6, False)])
def test_les_poles_sont_reels_jusqu_a_q_un_demi_compris(
    qualite: float, *, reels: bool
) -> None:
    """Le corrige de l'examen blanc ecarte `Q = 1/2` ; le pole y est double, reel."""
    poles = sp.roots(sp.Poly(s**2 + s / sp.Rational(str(qualite)) + 1, s))
    assert all(p.is_real for p in poles) is reels
    assert all(sp.re(p) < 0 for p in poles)


def test_le_courant_de_polarisation_amene_l_integrateur_a_rf_i0() -> None:
    """Article « Le courant de polarisation » : `R_f I_0 (1 - e^(-t / R_f C_f))`."""
    courant = sp.Symbol("I_0", positive=True)
    sortie = (1 / C) / (s + 1 / (R * C)) * courant / s
    temporel = sp.inverse_laplace_transform(sortie, s, t)
    attendu = R * courant * (1 - sp.exp(-t / (R * C)))
    assert sp.simplify(temporel - attendu) == 0
    assert sp.limit(temporel, t, sp.oo) == R * courant


def test_le_non_inverseur_a_condensateur_a_un_gain_continu_unite() -> None:
    """Article « Le montage non-inverseur », sur le sujet de janvier 2025."""
    r1, rf = sp.symbols("R_1 R_f", positive=True)
    transfert = 1 + rf / (r1 + 1 / (s * C))
    assert sp.simplify(transfert - (1 + s * C * (r1 + rf)) / (1 + s * C * r1)) == 0
    assert sp.limit(transfert, s, 0) == 1
    assert sp.limit(transfert, s, sp.oo) == 1 + rf / r1


def test_les_elements_du_prototype_de_butterworth() -> None:
    assert elements_de_butterworth(2) == pytest.approx((math.sqrt(2), math.sqrt(2)))
    assert elements_de_butterworth(3) == pytest.approx((1, 2, 1))


def test_passer_de_90_a_45_ohms_divise_les_bobines_et_double_les_condensateurs() -> (
    None
):
    resistance, bobine, condensateur = mise_a_l_echelle_d_impedance(0.5, 90, 1e-6, 1e-9)
    assert (resistance, bobine, condensateur) == pytest.approx((45, 0.5e-6, 2e-9))


@pytest.mark.parametrize(
    ("gain", "produit", "bande", "etages"),
    [(500, 2.5e6, 100e3, 2), (50, 2e6, 500e3, 3), (5000, 1e6, 50e3, 3)],
)
def test_le_nombre_d_etages_des_trois_sujets(
    gain: float, produit: float, bande: float, etages: int
) -> None:
    """Janvier 2025 (theorie, 3.1), janvier 2023 (4.1), janvier 2019 (2.1)."""
    assert nombre_d_etages(gain, produit, bande) == etages
    assert gain_maximal_par_etage(produit, bande) ** etages >= gain


def test_le_pas_d_un_convertisseur_de_douze_bits_entre_zero_et_quinze_volts() -> None:
    """Janvier 2025 (theorie, 1.3) : 3,66 mV, que l'on divise par 2^n ou 2^n - 1."""
    exact = float(Quantificateur(reference=7.5, bits=12).pas())
    assert exact == pytest.approx(3.663e-3, rel=1e-3)
    assert exact == pytest.approx(15 / 2**12, rel=3e-4)


def test_friis_porte_les_gains_en_tension_au_carre() -> None:
    assert facteur_en_cascade(2, 10, 4) == pytest.approx(2 + 9 / 16)
