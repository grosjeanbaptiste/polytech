"""Fiche d'electronique : capteur, bruit, facteur de bruit, Friis.

Articles 30 a 36 de `cursus/fiches/ElectroniqueInstrumentation.toml`.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.electronique.domain.acquisition import (
    densite_de_bruit_maximale,
    densite_spectrale_unilaterale_depuis_bilaterale,
    variance_d_un_bruit_borne,
)
from polytech.electronique.domain.bruit import (
    densite_spectrale_en_un_sur_f,
    puissance_integree_blanc_et_rose,
)
from polytech.electronique.domain.bruit_thermique import (
    densite_thermique,
    tension_efficace_thermique,
)
from polytech.electronique.domain.capteur import erreur_d_estimation, sensibilite
from polytech.electronique.domain.facteur_de_bruit import (
    TEMPERATURE_DE_REFERENCE as T0,
)
from polytech.electronique.domain.facteur_de_bruit import (
    facteur_de_bruit,
    figure_de_bruit,
)
from polytech.electronique.domain.friis import (
    bruit_ramene_a_l_entree,
    facteur_d_une_cascade,
)
from polytech.electronique.domain.modele_tension_courant import (
    ModeleBlancEtRose,
    densite_ramenee_a_l_entree,
    facteur_de_bruit_du_modele,
)
from polytech.maths.domain.comparaison import egales, proches
from polytech.signaux.domain.processus_aleatoires import densite_en_sortie
from polytech.thermique.domain.rayonnement.corps_noir import BOLTZMANN

X, N = sp.symbols("X N", positive=True)
K_B, T, R, C, F = sp.symbols("k_B T R C f", positive=True)


def test_le_bruit_deplace_l_estimation_de_bruit_sur_sensibilite() -> None:
    """Article « La sensibilite » : `S = f'(X)`, `dX ~ N / S`."""
    pente = float(sensibilite(X**2, X, 3))
    assert pente == 6
    bruit = 1e-4
    ecart = math.sqrt(9 + bruit) - 3
    assert proches(ecart, erreur_d_estimation(bruit, pente), 1e-8)


def test_la_densite_en_sortie_et_l_unilaterale_valent_le_double() -> None:
    """Article « La densite spectrale » : `|H|^2 S_XX`, unilaterale `2 S`."""
    h = 1 / (1 + sp.I * 2 * sp.pi * F * R * C)
    sortie = densite_en_sortie(h, sp.Integer(3))
    assert egales(sortie, 3 / (1 + (2 * sp.pi * F * R * C) ** 2))
    assert densite_spectrale_unilaterale_depuis_bilaterale(3.0) == 6.0
    assert variance_d_un_bruit_borne(6.0, 100.0) == 600.0


def test_le_bruit_thermique_et_son_filtrage_donnent_kt_sur_c() -> None:
    """Article « Le bruit thermique » : `4kTR`, `sqrt(4kTRB)`, `kT/C`."""
    densite = 4 * K_B * T * R
    total = sp.integrate(densite / (1 + (2 * sp.pi * F * R * C) ** 2), (F, 0, sp.oo))
    assert egales(total, K_B * T / C)
    assert proches(densite_thermique(1e3, 300.0), 4 * BOLTZMANN * 300 * 1e3, 1e-30)
    rms = tension_efficace_thermique(1e3, 300.0, 1e4)
    assert proches(rms, math.sqrt(densite_thermique(1e3, 300.0) * 1e4), 1e-15)


def test_le_bruit_rose_integre_un_logarithme_et_le_coin_egale_le_blanc() -> None:
    """Article « Le bruit rose » : `K/f`, coin `K/f_c = S_blanc`."""
    w, k, f1, f2 = sp.symbols("w K f_1 f_2", positive=True)
    integrale = sp.integrate(w + k / F, (F, f1, f2))
    assert egales(integrale, w * (f2 - f1) + k * sp.log(f2 / f1))
    assert proches(
        puissance_integree_blanc_et_rose(2.0, 3.0, 1.0, 10.0),
        18 + 3 * math.log(10),
        1e-12,
    )
    modele = ModeleBlancEtRose(blanc=4e-18, coin=50.0)
    assert proches(modele.coefficient_rose() / 50.0, 4e-18, 1e-30)
    k = modele.coefficient_rose()
    assert proches(densite_spectrale_en_un_sur_f(k, 50.0), 4e-18, 1e-30)


def test_le_bruit_ramene_a_l_entree_et_le_facteur_de_bruit_du_modele() -> None:
    """Articles « Le bruit ramene a l'entree » et « Le facteur de bruit »."""
    sv, si, rs, bande = (3e-9) ** 2, (1e-12) ** 2, 1e3, 1e4
    thermique = 4 * BOLTZMANN * T0 * rs
    assert proches(
        densite_ramenee_a_l_entree(rs, sv, si), thermique + sv + rs**2 * si, 1e-30
    )
    signal = 1e-6
    snr_entree = signal**2 / (thermique * bande)
    snr_sortie = signal**2 / ((thermique + sv + rs**2 * si) * bande)
    facteur = facteur_de_bruit(snr_entree, snr_sortie)
    attendu = facteur_de_bruit_du_modele(
        math.sqrt(sv * bande), math.sqrt(si * bande), rs, bande=bande
    )
    assert proches(facteur, attendu, 1e-9)
    assert proches(figure_de_bruit(2.0), 3.0103, 1e-4)


def test_friis_divise_chaque_etage_par_les_gains_qui_le_precedent() -> None:
    """Article « La formule de Friis » : `F1 + (F2-1)/A1^2 + (F3-1)/(A1 A2)^2`."""
    etages = ((2.0, 10.0), (4.0, 5.0), (9.0, 1.0))
    attendu = 2 + 3 / 100 + 8 / (100 * 25)
    assert proches(facteur_d_une_cascade(etages), attendu, 1e-12)
    excedents = tuple(((f - 1) * 1.0, a) for f, a in etages)
    assert proches(1 + bruit_ramene_a_l_entree(excedents), attendu, 1e-12)
    assert facteur_d_une_cascade(((1.0, 10.0), (1.0, 1.0))) == 1.0


def test_la_densite_maximale_se_deduit_du_bruit_total_sur_la_bande() -> None:
    """Janvier 2025 (7.5) : `e_n = V_bruit / sqrt(delta f)`, `53 nV/sqrt(Hz)`."""
    assert proches(densite_de_bruit_maximale(0.75e-6, 200.0), 53.03e-9, 1e-11)
    assert proches(
        densite_de_bruit_maximale(0.75e-6, 200.0) ** 2 * 200, (0.75e-6) ** 2, 1e-20
    )
