"""Le point d'aspiration d'une pompe : pression statique minimale et cavitation.

Oracles : Bernoulli depuis la surface libre (vitesse nulle, `p_atm`) ; la
pression de vapeur de l'eau a 20 degres C, 2,339 kPa (tables usuelles) ;
et la hauteur d'aspiration maximale `(p_atm - p_v) / (rho g)`, environ
10 m, que tout manuel de pompes donne.
"""

from __future__ import annotations

import pytest

from polytech.fluides.domain.cavitation import (
    PRESSION_DE_VAPEUR_EAU_20C,
    hauteur_d_aspiration_maximale,
    npsh_disponible,
    pression_dans_l_aspiration,
    risque_de_cavitation,
)
from polytech.fluides.domain.grandeurs import PESANTEUR

P_ATM, RHO = 101_325.0, 1000.0


def test_la_pression_de_vapeur_de_l_eau_a_20_degres() -> None:
    assert PRESSION_DE_VAPEUR_EAU_20C == 2339.0


def test_au_repos_a_la_surface_la_pression_est_atmospherique() -> None:
    assert pression_dans_l_aspiration(P_ATM, RHO, 0.0, 0.0, 0.0) == P_ATM


def test_a_deux_metres_au_dessus_de_la_surface_la_pression_chute_de_rho_g_h() -> None:
    p = pression_dans_l_aspiration(P_ATM, RHO, 2.0, 0.0, 0.0)
    assert p == pytest.approx(P_ATM - RHO * PESANTEUR * 2.0)


def test_la_vitesse_et_les_pertes_abaissent_encore_la_pression() -> None:
    p = pression_dans_l_aspiration(P_ATM, RHO, 2.0, 1.5, 3000.0)
    assert p == pytest.approx(P_ATM - RHO * PESANTEUR * 2 - RHO * 1.5**2 / 2 - 3000.0)


def test_une_pompe_sous_la_surface_est_en_surpression() -> None:
    assert pression_dans_l_aspiration(P_ATM, RHO, -1.0, 0.0, 0.0) > P_ATM


def test_une_perte_de_charge_negative_leve() -> None:
    with pytest.raises(ValueError, match="perte"):
        pression_dans_l_aspiration(P_ATM, RHO, 2.0, 1.0, -1.0)


def test_une_masse_volumique_non_positive_leve() -> None:
    with pytest.raises(ValueError, match="masse volumique"):
        pression_dans_l_aspiration(P_ATM, 0.0, 2.0, 1.0, 0.0)


def test_la_cavitation_commence_quand_la_pression_atteint_la_vapeur() -> None:
    assert risque_de_cavitation(PRESSION_DE_VAPEUR_EAU_20C, PRESSION_DE_VAPEUR_EAU_20C)
    assert risque_de_cavitation(1000.0, PRESSION_DE_VAPEUR_EAU_20C)
    assert not risque_de_cavitation(2340.0, PRESSION_DE_VAPEUR_EAU_20C)


def test_une_pression_calculee_negative_est_une_cavitation_et_non_une_erreur() -> None:
    p = pression_dans_l_aspiration(P_ATM, RHO, 12.0, 0.0, 0.0)
    assert p < 0
    assert risque_de_cavitation(p, PRESSION_DE_VAPEUR_EAU_20C)


def test_une_pression_de_vapeur_negative_leve() -> None:
    with pytest.raises(ValueError, match="vapeur"):
        risque_de_cavitation(1e5, -1.0)


def test_la_hauteur_d_aspiration_maximale_de_l_eau_est_d_environ_dix_metres() -> None:
    h = hauteur_d_aspiration_maximale(P_ATM, PRESSION_DE_VAPEUR_EAU_20C, RHO, 0.0, 0.0)
    assert h == pytest.approx((P_ATM - 2339.0) / (RHO * PESANTEUR))
    assert 10.0 < h < 10.2


def test_a_la_hauteur_maximale_la_pression_atteint_exactement_la_vapeur() -> None:
    h = hauteur_d_aspiration_maximale(P_ATM, 2339.0, RHO, 1.2, 4000.0)
    p = pression_dans_l_aspiration(P_ATM, RHO, h, 1.2, 4000.0)
    assert p == pytest.approx(2339.0)


def test_les_pertes_reduisent_la_hauteur_d_aspiration_maximale() -> None:
    sans = hauteur_d_aspiration_maximale(P_ATM, 2339.0, RHO, 1.0, 0.0)
    avec = hauteur_d_aspiration_maximale(P_ATM, 2339.0, RHO, 1.0, 5000.0)
    assert avec < sans


def test_le_npsh_disponible_se_lit_depuis_la_pression_a_l_entree() -> None:
    p = pression_dans_l_aspiration(P_ATM, RHO, 2.0, 1.5, 3000.0)
    attendu = (P_ATM - 2339.0) / (RHO * PESANTEUR) - 2.0 - 3000.0 / (RHO * PESANTEUR)
    assert npsh_disponible(p, 1.5, 2339.0, RHO) == pytest.approx(attendu)


def test_le_npsh_vaut_la_hauteur_cinetique_quand_la_statique_touche_la_vapeur() -> None:
    h = hauteur_d_aspiration_maximale(P_ATM, 2339.0, RHO, 1.0, 1000.0)
    p = pression_dans_l_aspiration(P_ATM, RHO, h, 1.0, 1000.0)
    assert npsh_disponible(p, 1.0, 2339.0, RHO) == pytest.approx(1.0 / (2 * PESANTEUR))


def test_le_npsh_est_nul_a_vitesse_nulle_quand_la_pression_touche_la_vapeur() -> None:
    assert npsh_disponible(2339.0, 0.0, 2339.0, RHO) == 0.0


def test_le_npsh_leve_pour_une_masse_volumique_non_positive() -> None:
    with pytest.raises(ValueError, match="masse volumique"):
        npsh_disponible(1e5, 1.0, 2339.0, 0.0)
