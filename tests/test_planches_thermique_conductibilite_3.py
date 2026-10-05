"""Les planches de thermique : la conductibilite non stationnaire.

Source : `ThermiqueFondements/SLIDES/Slides_04_Conductibilite_3.pdf`, les
14 planches, lues une a une.

Les oracles : le nombre de Biot et son seuil de la planche 3, la reponse
du premier ordre de la planche 5, l'abaque de la planche 9, les deux
temperatures de contact de la planche 13 et la bille de la planche 14.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.thermique.domain.conduction.plaque_variable import PlaqueEnEchelon
from polytech.thermique.domain.conduction.semi_infini import (
    sous_un_echelon_de_surface,
    temperature_de_contact,
)
from polytech.thermique.domain.inertie import SolideIsotherme
from polytech.thermique.domain.transitoire import CorpsTransitoire

ACIER = (46.5, 7850.0 * 460.4)
"""Planche 14 : la conductivite et la capacite volumique de l'acier."""


def test_le_nombre_de_biot_separe_les_deux_cas() -> None:
    """Planche 3 : `Bi = K e / lambda = (e / lambda) / (1 / K)`, mince sous 0,1."""
    bille = CorpsTransitoire(10.0, 0.05 / 6, *ACIER)
    assert bille.biot() == pytest.approx((0.05 / 6 / 46.5) / (1 / 10.0))
    assert bille.est_thermiquement_mince()
    trempee = CorpsTransitoire(5000.0, 0.05 / 6, *ACIER)
    assert trempee.biot() > 0.1
    assert not trempee.est_thermiquement_mince()
    with pytest.raises(ValueError, match="corps non thermiquement mince"):
        trempee.constante_de_temps()


def test_la_constante_de_temps_est_un_produit_r_c() -> None:
    """Planche 4 : `tau_c = rho c V / (K S) = (1 / K S) m c = R C`."""
    surface, volume = math.pi * 0.05**2, math.pi * 0.05**3 / 6
    solide = SolideIsotherme.depuis(
        echange=10.0, surface=surface, capacite=ACIER[1] * volume
    )
    attendue = ACIER[1] * volume / (10.0 * surface)
    assert float(solide.constante_de_temps()) == pytest.approx(attendue)
    bille = CorpsTransitoire(10.0, 0.05 / 6, *ACIER)
    assert bille.constante_de_temps() == pytest.approx(attendue)


def test_la_reponse_du_premier_ordre_passe_a_63_pour_cent() -> None:
    """Planche 5 : `(t - t_0)/(t_f - t_0) = 1 - exp(-tau/tau_c)`, 0,632 en `tau_c`."""
    solide = SolideIsotherme(resistance=2.0, capacite=150.0)
    monte = 1 - float(solide.ecart_reduit(300.0))
    assert monte == pytest.approx(0.632, abs=5e-4)
    duree = sp.Symbol("tau", positive=True)
    pente = sp.diff(1 - solide.ecart_reduit(duree), duree).subs(duree, 0)
    assert float(pente) == pytest.approx(1 / 300)


def test_biot_et_fourier_sont_des_rapports_de_la_plaque() -> None:
    """Planche 10 : `Bi = R_i / R_e`, `Fo = tau / tau_c` avec `tau_c = R_i C`."""
    echange, demi_epaisseur = 400.0, 0.02
    plaque = CorpsTransitoire(echange, demi_epaisseur, *ACIER)
    interne, externe = demi_epaisseur / ACIER[0], 1 / echange
    assert plaque.biot() == pytest.approx(interne / externe)
    capacite = ACIER[1] * demi_epaisseur
    assert plaque.fourier(30.0) == pytest.approx(30.0 / (interne * capacite))


@pytest.mark.parametrize(("inverse_de_biot", "fourier"), [(0.1, 1.24)])
def test_l_abaque_de_la_planche_9_au_centre(
    inverse_de_biot: float, fourier: float
) -> None:
    """`x/L = 0` : la courbe `Bi^-1 = 0,1` coupe 0,1 en `Fo = 1,24`, lu au pixel."""
    plaque = PlaqueEnEchelon(1 / inverse_de_biot)
    assert plaque.ecart_reduit(fourier, 0.0) == pytest.approx(0.1, abs=0.003)


def test_l_abaque_tend_vers_la_surface_imposee() -> None:
    """La courbe `Bi^-1 = 0` coupe 0,1 vers `Fo = 1,0` : `(4/pi) exp(-(pi/2)^2 Fo)`."""
    plaque = PlaqueEnEchelon(1e6)
    limite = 4 / math.pi * math.exp(-((math.pi / 2) ** 2) * 1.03)
    assert plaque.ecart_reduit(1.03, 0.0) == pytest.approx(limite, rel=1e-3)
    assert limite == pytest.approx(0.1, abs=0.001)
    assert plaque.ecart_reduit(1.0, 0.0) == pytest.approx(0.1, abs=0.01)


def test_la_surface_refroidit_avant_le_centre() -> None:
    """Planche 8 : a temps court, peu d'effet au centre, refroidissement en surface."""
    plaque = PlaqueEnEchelon(4.0)
    assert plaque.ecart_reduit(0.05, 0.0) > 0.99
    assert plaque.ecart_reduit(0.05, 1.0) < 0.6


def test_a_temps_court_la_plaque_est_un_solide_semi_infini() -> None:
    """Planche 11 : sous un echelon de surface, `erf(x / (2 sqrt(a tau)))`."""
    x, duree, a = sp.symbols("x tau a", positive=True)
    champ = sous_un_echelon_de_surface(x, duree, a, initiale=0, surface=1)
    assert sp.simplify(sp.diff(champ, duree) - a * sp.diff(champ, x, 2)) == 0
    assert sp.limit(champ, x, 0) == 1
    assert sp.limit(champ, x, sp.oo) == 0
    plaque = PlaqueEnEchelon(1e6)
    fourier, depuis_la_surface = 0.01, 0.1
    semi_infini = float(champ.subs({x: depuis_la_surface, duree: fourier, a: 1}))
    dans_la_plaque = 1 - plaque.ecart_reduit(fourier, 1 - depuis_la_surface, 200)
    assert dans_la_plaque == pytest.approx(semi_infini, abs=1e-3)


@pytest.mark.parametrize(("effusivite", "ressentie"), [(14000.0, 21.5), (400.0, 33.2)])
def test_la_temperature_ressentie_de_la_planche_13(
    effusivite: float, ressentie: float
) -> None:
    """`t_m = (t_1 b_1 + t_2 b_2)/(b_1 + b_2)` : l'acier, puis le bois, sous la main."""
    contact = temperature_de_contact((20.0, effusivite**2), (37.0, 1400.0**2))
    assert contact == pytest.approx(ressentie, abs=0.06)


def test_l_interface_est_du_cote_de_la_plus_forte_effusivite() -> None:
    """Planche 12 : `(t_1 - t_m)/(t_m - t_2) = b_2 / b_1`, `b = sqrt(lambda rho c)`."""
    acier = math.sqrt(46.5 * 7850.0 * 460.4)
    assert acier == pytest.approx(12960.0, rel=1e-3)
    contact = temperature_de_contact((600.0, acier**2), (20.0, 1400.0**2))
    assert (600.0 - contact) / (contact - 20.0) == pytest.approx(1400.0 / acier)
    assert contact > 500.0


def test_la_bille_d_acier_de_la_planche_14() -> None:
    """De 600 a 100 degres dans une ambiance a 60, `K = 10` : `tau_c ln(13,5)`."""
    bille = CorpsTransitoire(10.0, 0.05 / 6, *ACIER)
    constante = bille.constante_de_temps()
    assert constante == pytest.approx(3011.8, abs=0.5)
    duree = constante * math.log((600.0 - 60.0) / (100.0 - 60.0))
    assert duree == pytest.approx(7839.0, abs=3.0)
    assert bille.ecart_relatif(duree) == pytest.approx(40.0 / 540.0)
