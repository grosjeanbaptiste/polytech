"""Profils de temperature d'un echangeur a courants paralleles, et vitre selective."""

from __future__ import annotations

import math

import pytest

from polytech.maths.domain.comparaison import proches, proches_relativement
from polytech.thermique.domain.echangeurs import moyenne_logarithmique
from polytech.thermique.domain.echangeurs_a_courants_paralleles import Sens
from polytech.thermique.domain.profils_d_echangeur import profils_cocourants
from polytech.thermique.domain.rayonnement.vitre_selective import VitreSelective
from polytech.thermique.domain.rendement_d_echangeur import rendement

ENTREES = (90.0, 20.0)


def test_les_extremites_donnent_le_rendement_du_cocourant() -> None:
    """`(t_1E - t_1S)/(t_1E - t_2E) = (1 - e^(-N(1+R)))/(1+R)`."""
    chaud, _ = profils_cocourants(*ENTREES, unites=1.2, rapport=0.5, position=1.0)
    assert proches_relativement(
        (90.0 - chaud) / 70.0, rendement(Sens.COCOURANT, 1.2, 0.5), 1e-12
    )


def test_le_bilan_et_la_moyenne_logarithmique_sont_coherents() -> None:
    """`C_1 (t_1E - t_1S) = K S dt_log`, avec `N = K S / C_1`."""
    unites, c1 = 0.9, 100.0
    chaud, froid = profils_cocourants(*ENTREES, unites=unites, rapport=1.0, position=1.0)
    ecart_b = chaud - froid
    flux = c1 * (90.0 - chaud)
    assert proches_relativement(flux, unites * c1 * moyenne_logarithmique(70.0, ecart_b), 1e-9)


def test_a_l_entree_les_temperatures_sont_celles_de_l_entree() -> None:
    """`x = 0` : rien n'a ete echange."""
    assert profils_cocourants(*ENTREES, unites=1.0, rapport=1.0, position=0.0) == ENTREES


def test_une_position_hors_de_l_echangeur_est_refusee() -> None:
    """`x` hors de `[0, 1]` : refuse."""
    with pytest.raises(ValueError, match="position"):
        profils_cocourants(*ENTREES, unites=1.0, rapport=1.0, position=1.5)


def test_le_fluide_chaud_se_refroidit_et_le_froid_s_echauffe_de_concert() -> None:
    """Bilan : `C_1 dt_1 = -C_2 dt_2`, soit `dt_2 = R dt_1` en module."""
    chaud, froid = profils_cocourants(*ENTREES, unites=1.0, rapport=0.5, position=0.7)
    assert proches(froid - 20.0, 0.5 * (90.0 - chaud), 1e-12)
    assert math.isfinite(chaud)


def test_la_vitre_selective_transmet_sous_la_coupure_et_pas_au_dela() -> None:
    """Exercice de la serre : `tau_lambda = 0,94` sous 3,5 um, nul au-dela."""
    verre = VitreSelective(0.94, 0.06, 3.5e-6)
    assert verre.transmission_spectrale(1e-6) == 0.94
    assert verre.transmission_spectrale(5e-6) == 0.0
