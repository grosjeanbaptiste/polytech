"""Fiche d'electronique : convertisseur, multiplexeur, chaine, boitiers, induction.

Articles 10, 11 et 37 a 40 de `cursus/fiches/ElectroniqueInstrumentation.toml`.
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.biot_et_savart import (
    integrale_de_biot_et_savart,
)
from polytech.electronique.domain.acquisition import (
    cadence_du_multiplexeur,
    gain_de_chaine,
    nyquist,
)
from polytech.electronique.domain.amplification import gain_maximal_par_etage
from polytech.electronique.domain.code_des_composants import (
    numero_oppose_sur_un_dip,
)
from polytech.electronique.domain.induction import BoucleDeMesure
from polytech.electronique.domain.multiplexage import (
    paires_multiplexeur_convertisseur,
)
from polytech.electronique.domain.quantification import Quantificateur
from polytech.maths.domain.comparaison import egales, proches

MU0, COURANT, OMEGA, T = sp.symbols("mu_0 I omega t", positive=True)
A, B, L = sp.symbols("a b ell", positive=True)


def test_le_flux_d_un_fil_infini_porte_ln_de_a_plus_b_sur_a() -> None:
    """Article « Un flux variable » : `mu_0 I l / (2 pi) ln((a+b)/a)`."""
    boucle = BoucleDeMesure(hauteur=L, distance=A, largeur=B)
    flux = boucle.flux(MU0, COURANT, OMEGA, T)
    amplitude = MU0 * COURANT * L / (2 * sp.pi) * sp.log((A + B) / A)
    assert egales(flux, amplitude * sp.sin(OMEGA * T))
    champ_au_bord = sp.diff(flux, B) / (L * sp.sin(OMEGA * T))
    assert egales(champ_au_bord, MU0 * COURANT / (2 * sp.pi * (A + B)))
    avec = {A: 1, B: 3, MU0: 1, COURANT: 1, L: 1}
    juste, faux = sp.log(sp.Integer(4)), sp.log(sp.Rational(4, 3))
    assert not egales(juste, faux)
    assert egales(amplitude.subs(avec), juste / (2 * sp.pi))


def test_jefimenko_statique_redonne_biot_et_savart_au_centre_d_une_spire() -> None:
    """Article « De Jefimenko a Biot-Savart » : `dJ/dt = 0`, `B = mu_0 I/(2a)`."""
    rayon, angle = 2.0, sp.Symbol("phi")
    cercle = (rayon * sp.cos(angle), rayon * sp.sin(angle), sp.Integer(0))
    _, _, axiale = integrale_de_biot_et_savart(
        cercle, angle, 0.0, 2 * math.pi, (0, 0, 0)
    )
    assert proches(axiale / (4 * math.pi), 1 / (2 * rayon), 1e-9)
    assert sp.diff(COURANT, T) == 0


def test_le_pas_du_convertisseur_est_la_pleine_echelle_sur_2_puissance_n_moins_1() -> (
    None
):
    """Article « Choisir le convertisseur » : `LSB = FS/(2^n - 1)`, `f_s >= 2 f_max`."""
    cna = Quantificateur(reference=7.5, bits=12)
    assert cna.code_maximal() == 4095
    assert proches(cna.pas(), 15 / 4095, 1e-15)
    assert round(cna.pas() * 1e3, 2) == round(15 / 4096 * 1e3, 2) == 3.66
    gain = gain_de_chaine(15.0, 0.05)
    assert proches(cna.pas() / gain, 15 / 4095 / (15 / 0.05), 1e-18)
    assert nyquist(400.0) == 800.0


def test_le_multiplexeur_commute_m_fois_la_cadence_et_les_paires_se_comptent() -> None:
    """Article « Le multiplexeur » : 8 voies a 400 Hz, `ceil(capteurs / m)`."""
    assert cadence_du_multiplexeur(400.0, 8) == 3200.0
    assert cadence_du_multiplexeur(400.0, 8) > 2000.0
    assert paires_multiplexeur_convertisseur(45, 8) == math.ceil(45 / 8) == 6


def test_le_gain_du_premier_etage_est_borne_par_saturation_et_gbw() -> None:
    """Janvier 2023 (10.7) : `5 V / 40 mV = 125` ; borne `GBW / B`."""
    saturation = gain_de_chaine(5.0, 40e-3)
    assert proches(saturation, 125.0, 1e-9)
    borne = gain_maximal_par_etage(1e6, 5e3)
    assert min(saturation, borne) == 125.0
    assert min(saturation, gain_maximal_par_etage(1e6, 50e3)) == 20.0


def test_la_patte_opposee_d_un_boitier_est_n_plus_un_moins_k() -> None:
    """Article « Cabler un boitier » : `k <-> N + 1 - k`."""
    for pattes in (8, 14, 16):
        for k in range(1, pattes + 1):
            assert numero_oppose_sur_un_dip(k, pattes) == pattes + 1 - k


def test_la_figure_du_convertisseur_ecrit_le_meme_pas_que_l_article() -> None:
    """Article « Choisir le convertisseur » : `2^n - 1` partout, jamais `2^n`."""
    chemin = Path(__file__).resolve().parent.parent / "cursus" / "fiches"
    fiche = tomllib.loads(
        (chemin / "ElectroniqueInstrumentation.toml").read_text("utf-8")
    )
    notion = "echantillonnage_et_quantification"
    article = next(a for a in fiche["article"] if a["notion"] == notion)
    assert "2^n - 1" in article["formule"]
    assert "2^n - 1" in article["figure"]
