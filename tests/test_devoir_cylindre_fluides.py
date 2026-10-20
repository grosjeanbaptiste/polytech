"""Devoir cote de fluides : les fluides, le Reynolds, les regimes, le Mach.

Chaque donnee vient d'un document du depot, et `test_sources_devoir_cylindre`
retrouve chaque passage cite dans le texte converti :

- l'air a 20 degres de l'introduction du cours de fluides ;
- l'air a une autre temperature par la correlation du TP 4 de thermique,
  sa masse volumique par le gaz parfait ;
- l'eau de l'activite d'integration 1 et du TP 4 ;
- les regimes du cylindre isole : la planche 31 de la convection forcee
  (thermique) et la courbe `C_D(Re)` de la planche 32 d'aerodynamique.
"""

from __future__ import annotations

import math

import pytest

from polytech.fluides.application import devoir_cylindre_fluides as dcf
from polytech.thermique.application.proprietes_tp_04 import viscosite_de_l_air


def test_l_air_a_vingt_degres_est_celui_de_l_introduction() -> None:
    """`mu = 1,83e-5 kg/(m s)` et `nu = 1,52e-5 m^2/s` a 20 degres, 1 atm."""
    assert dcf.AIR.viscosite == pytest.approx(1.83e-5)
    assert dcf.AIR.viscosite_cinematique == pytest.approx(1.52e-5, rel=0.01)


def test_l_air_a_une_autre_temperature_suit_la_correlation_du_tp_4() -> None:
    froid = dcf.air_a(-10.0)
    assert froid.viscosite == pytest.approx(viscosite_de_l_air(-10.0))
    assert froid.masse_volumique == pytest.approx(
        dcf.AIR.masse_volumique * 293.15 / 263.15
    )


def test_les_deux_sources_de_l_air_s_accordent_a_vingt_degres() -> None:
    """Le cours de fluides et la correlation du TP 4 : moins de deux pour cent."""
    assert dcf.air_a(20.0).viscosite == pytest.approx(dcf.AIR.viscosite, rel=0.02)


def test_l_eau_a_vingt_degres() -> None:
    """`rho = 1000`, `mu = 27,531e-6 exp(527,93 / (126,53 + 20))` Pa s."""
    assert dcf.EAU.masse_volumique == pytest.approx(1000.0)
    attendu = 27.531e-6 * math.exp(527.93 / 146.53)
    assert dcf.EAU.viscosite == pytest.approx(attendu)
    assert dcf.EAU.celerite is None


def test_le_reynolds_du_cylindre() -> None:
    """`Re = rho U D / mu`, le diametre pour longueur."""
    attendu = dcf.AIR.masse_volumique * 10 * 0.5 / dcf.AIR.viscosite
    assert dcf.reynolds(dcf.AIR, 10.0, 0.5) == pytest.approx(attendu)


@pytest.mark.parametrize(
    ("reynolds", "regime"),
    [
        (0.5, dcf.Regime.RAMPANT),
        (1.0, dcf.Regime.LAMINAIRE),
        (9.9e4, dcf.Regime.LAMINAIRE),
        (1e5, dcf.Regime.TRANSITION),
        (4.9e5, dcf.Regime.TRANSITION),
        (5e5, dcf.Regime.TURBULENT),
    ],
)
def test_le_regime_suit_les_deux_planches(reynolds: float, regime: dcf.Regime) -> None:
    assert dcf.regime(reynolds) is regime


def test_le_decollement_laminaire_et_turbulent() -> None:
    """Planche 31 de thermique : vers 80 degres, puis vers 110 apres transition."""
    assert dcf.Regime.LAMINAIRE.decollement == 80
    assert dcf.Regime.TURBULENT.decollement == 110
    assert dcf.Regime.RAMPANT.decollement is None


def test_un_reynolds_negatif_leve() -> None:
    with pytest.raises(ValueError, match="nombre de Reynolds negatif"):
        dcf.regime(-1.0)


def test_la_celerite_du_son_depend_de_la_temperature() -> None:
    """`c = racine(gamma R T / M)` : environ 343 m/s a 20 degres, 325 a -10."""
    assert dcf.AIR.celerite == pytest.approx(343.0, abs=0.5)
    assert dcf.air_a(-10.0).celerite == pytest.approx(325.0, abs=0.5)


def test_le_mach_de_cent_trois_metres_par_seconde_est_le_seuil() -> None:
    """Le cours : Ma = 0,3, soit 103 m/s dans l'air."""
    assert dcf.mach(dcf.AIR, 103.0) == pytest.approx(0.3, abs=0.002)
    assert dcf.mach(dcf.EAU, 1.0) is None
