"""Les cas particuliers de la fiche d'electronique, refaits par deux chemins.

Les cas que `tests/test_fiche_d_electronique.py` refait deja (le nombre
d'etages de trois sujets, l'admittance d'entree, Friis, la mise a l'echelle,
le courant de polarisation) n'y sont pas repetes : ce fichier prend les
autres.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.electronique.domain.amplification import (
    gain_maximal_par_etage,
    nombre_d_etages,
)


def test_seconde_session_2019_cinq_etages_de_dix() -> None:
    """0,5 mV vers 10 V sur 200 kHz avec 2 MHz."""
    gain = 10 / 0.5e-3
    assert nombre_d_etages(gain, 2e6, 200e3) == 5
    assert gain_maximal_par_etage(2e6, 200e3) ** 4 < gain


def test_janvier_2020_la_charge_de_200_ohms_demande_60_milliamperes() -> None:
    assert math.isclose(12 / 200, 0.060)


def test_examen_blanc_le_diviseur_d_entree_divise_le_gain_par_deux() -> None:
    """Source de 1 kOhm, inverseur -5 d'entree 1 kOhm, puis non-inverseur 10."""
    source, entree = 1e3, 1e3
    avec_diviseur = entree / (source + entree) * (-5) * 10
    assert avec_diviseur == -25
    assert -5 * 10 == -50  # non-inverseur en tete : plus de diviseur


def test_aout_2022_le_bruit_thermique_filtre_par_un_rc_vaut_kt_sur_c() -> None:
    """Int |H|^2 4 kB T R df = kB T / C, quelle que soit R."""
    f, r, c, kt = sp.symbols("f R C kT", positive=True)
    densite = 4 * kt * r / (1 + (2 * sp.pi * f * r * c) ** 2)
    assert sp.simplify(sp.integrate(densite, (f, 0, sp.oo)) - kt / c) == 0


def test_janvier_2023_le_gain_du_premier_etage_est_borne_par_125() -> None:
    assert math.isclose(5 / 40e-3, 125)


def test_janvier_2025_huit_canaux_a_400_hertz_commutent_a_3200() -> None:
    assert 8 * 400 == 3200


def test_janvier_2025_le_non_inverseur_a_condensateur_et_ses_deux_limites() -> None:
    s, rf, r1, c = sp.symbols("s R_f R_1 C", positive=True)
    h = 1 + rf / (r1 + 1 / (s * c))
    assert sp.limit(h, s, 0) == 1
    assert sp.simplify(sp.limit(h, s, sp.oo) - (1 + rf / r1)) == 0


def test_janvier_2024_un_facteur_un_sur_s_integre_la_reponse() -> None:
    s, t, g, a = sp.symbols("s t G alpha", positive=True)
    impulsion = sp.inverse_laplace_transform(g / (s + a), s, t)
    echelon = sp.inverse_laplace_transform(g / (s * (s + a)), s, t)
    u = sp.Symbol("u", positive=True)
    assert sp.simplify(echelon - sp.integrate(impulsion.subs(t, u), (u, 0, t))) == 0
