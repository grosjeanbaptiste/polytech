"""Le solenoide a noyau, et l'auto-inductance d'une boucle carree.

Oracles : l'inductance se retrouve par le flux capte -- N spires,
chacune traversee par `B A` avec `B = mu N I / l` ; le courant de
saturation redonne, par `H = N I / l`, le champ de saturation ; et a la
frequence de croisement, `|j omega L|` vaut exactement `R`.
"""

from __future__ import annotations

import math

import pytest

from polytech.electronique.domain.solenoide import (
    courant_de_saturation,
    frequence_de_croisement,
    inductance_de_rosa,
    inductance_du_solenoide,
    permeabilite_lineaire,
)


def test_l_inductance_est_le_flux_capte_par_ampere() -> None:
    mu, spires, aire, longueur, courant = 1e-3, 20, 25e-6, 0.02, 0.1
    flux_capte = spires * (mu * spires * courant / longueur) * aire
    assert inductance_du_solenoide(
        mu, spires, aire, longueur
    ) * courant == pytest.approx(flux_capte)


def test_le_courant_de_saturation_produit_le_champ_de_saturation() -> None:
    courant = courant_de_saturation(160.0, 0.02, 20)
    assert 20 * courant / 0.02 == pytest.approx(160.0)


def test_la_permeabilite_est_la_pente_de_la_partie_lineaire() -> None:
    assert permeabilite_lineaire(0.15, 160.0) == pytest.approx(9.375e-4)


def test_une_pente_sur_un_champ_nul_est_refusee() -> None:
    with pytest.raises(ValueError, match="champ nul"):
        permeabilite_lineaire(0.15, 0.0)


def test_rosa_pour_la_boucle_de_cinq_centimetres() -> None:
    """`mu0/4pi 8a (ln(a/rho) + rho/a - 0,524)` : 153 nH."""
    assert inductance_de_rosa(0.05, 0.65e-3) == pytest.approx(153.3e-9, abs=0.1e-9)


def test_un_fil_plus_gros_que_la_boucle_est_refuse() -> None:
    with pytest.raises(ValueError, match="rayon"):
        inductance_de_rosa(0.001, 0.002)


def test_au_croisement_l_inductance_vaut_la_resistance() -> None:
    inductance, resistance = 153e-9, 10.0
    frequence = frequence_de_croisement(resistance, inductance)
    assert 2 * math.pi * frequence * inductance == pytest.approx(resistance)
