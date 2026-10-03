"""Trois surfaces dont une adiabatique : l'echange effectif entre les deux autres.

Une paroi adiabatique ne garde rien : elle reradie tout ce qu'elle recoit,
et sert de relais entre les deux surfaces actives. Dans le schema en
resistances, elle ajoute une branche en parallele de l'echange direct.

Oracles : deux plans infinis -- sans paroi, l'echange gris redonne la
formule tabulee `1/(1/eps1 + 1/eps2 - 1)` ; et une paroi vue egalement des
deux surfaces prend pour emittance la moyenne de leurs radiosites.
"""

from __future__ import annotations

import pytest

from polytech.thermique.domain.rayonnement.corps_noir import emittance
from polytech.thermique.domain.rayonnement.paroi_adiabatique import (
    EnceinteAParoiAdiabatique,
)
from polytech.thermique.domain.rayonnement.surfaces_grises import (
    flux_entre_deux_plans_gris,
)


def _enceinte(s13: float, s23: float) -> EnceinteAParoiAdiabatique:
    return EnceinteAParoiAdiabatique(
        aire_1=2.0,
        emissivite_1=0.6,
        temperature_1=1000.0,
        aire_2=2.0,
        emissivite_2=0.5,
        temperature_2=300.0,
        echange_12=2.0,
        echange_13=s13,
        echange_23=s23,
    )


def test_sans_paroi_on_retrouve_deux_plans_gris() -> None:
    enceinte = _enceinte(0.0, 0.0)
    attendu = flux_entre_deux_plans_gris(0.6, 0.5, 1000.0, 300.0, 2.0)
    assert enceinte.flux() == pytest.approx(attendu)


def test_la_paroi_ajoute_un_chemin_en_parallele() -> None:
    noir = _enceinte(0.5, 0.25).echange_noir()
    assert noir == pytest.approx(2.0 + 1 / (1 / 0.5 + 1 / 0.25))


def test_une_paroi_symetrique_prend_la_moyenne_des_radiosites() -> None:
    enceinte = _enceinte(0.4, 0.4)
    j1, j2 = enceinte.radiosites()
    assert emittance(enceinte.temperature_de_la_paroi()) == pytest.approx((j1 + j2) / 2)


def test_le_flux_traverse_les_resistances_de_surface() -> None:
    """`phi = eps1 S1 / rho1 (sigma T1^4 - J1)` : le flux sort par la surface 1."""
    enceinte = _enceinte(0.5, 0.25)
    j1, _ = enceinte.radiosites()
    sortant = 0.6 * 2.0 / 0.4 * (emittance(1000.0) - j1)
    assert sortant == pytest.approx(enceinte.flux())


def test_une_emissivite_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="emissivite"):
        EnceinteAParoiAdiabatique(
            aire_1=2.0,
            emissivite_1=0.0,
            temperature_1=1000.0,
            aire_2=2.0,
            emissivite_2=0.5,
            temperature_2=300.0,
            echange_12=2.0,
            echange_13=0.0,
            echange_23=0.0,
        ).flux()


def test_le_flux_entre_dans_la_surface_2_par_sa_resistance() -> None:
    """`phi = eps2 S2 / rho2 (J2 - sigma T2^4)` : le meme flux, de l'autre cote."""
    enceinte = _enceinte(0.5, 0.25)
    _, j2 = enceinte.radiosites()
    entrant = 0.5 * 2.0 / 0.5 * (j2 - emittance(300.0))
    assert entrant == pytest.approx(enceinte.flux())


def test_la_paroi_rend_tout_ce_qu_elle_recoit() -> None:
    """Bilan au noeud de la paroi : ce qui arrive par 13 repart par 23."""
    enceinte = _enceinte(0.5, 0.25)
    j1, j2 = enceinte.radiosites()
    paroi = emittance(enceinte.temperature_de_la_paroi())
    assert 0.5 * (j1 - paroi) == pytest.approx(0.25 * (paroi - j2))
