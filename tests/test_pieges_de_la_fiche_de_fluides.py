"""Les pieges et enonces qualitatifs de la fiche de fluides, calcules par la lib.

Bernoulli (point d'aspiration, deux instants, pertes), crise de trainee
de la balle de golf, frottement et rugosite, Boussinesq, surfaces de
reference, Reynolds et Froude comme rapports de forces.
"""

from __future__ import annotations

import pytest

from polytech.fluides.domain.aerodynamique import Ecoulement, SurfacePortante
from polytech.fluides.domain.cavitation import pression_dans_l_aspiration
from polytech.fluides.domain.cinematique.symboles import ESPACE
from polytech.fluides.domain.conduite import perte_de_charge_reguliere
from polytech.fluides.domain.fondamentales.bernoulli_instationnaire import (
    pression_en_aval_instationnaire,
    terme_instationnaire,
)
from polytech.fluides.domain.fondamentales.continuite import vitesse_dans_la_section
from polytech.fluides.domain.fondamentales.contraintes import contraintes_visqueuses
from polytech.fluides.domain.grandeurs import PESANTEUR, MasseVolumique
from polytech.fluides.domain.loi_de_puissance import vitesse_en_loi_de_puissance
from polytech.fluides.domain.pertes_de_charge import coefficient_de_colebrook
from polytech.fluides.domain.regime_de_reynolds import nombre_de_reynolds
from polytech.fluides.domain.similitude import nombre_de_froude
from polytech.fluides.domain.trainee_de_la_sphere import (
    REYNOLDS_CRITIQUE_BALLE_DE_GOLF,
    REYNOLDS_CRITIQUE_SPHERE_LISSE,
    coefficient_de_trainee_de_la_sphere,
    est_apres_la_crise_de_trainee,
)
from polytech.fluides.domain.viscosite_turbulente import tenseur_de_boussinesq
from polytech.fluides.domain.zone_rugueuse import (
    coefficient_en_zone_rugueuse,
    est_completement_rugueux,
)
from polytech.maths.domain.comparaison import proches, proches_relativement

RHO = 1030.0


def test_le_point_d_aspiration_demande_les_pertes_que_l_enonce_ne_donne_pas() -> None:
    sans = pression_dans_l_aspiration(1.0e5, RHO, 2.0, 3.0, 0.0)
    avec = pression_dans_l_aspiration(1.0e5, RHO, 2.0, 3.0, 5.0e3)
    assert proches(sans - avec, 5.0e3, 1e-9)
    assert not proches(avec, 1.0e5, 1.0)


def test_bernoulli_ne_relie_que_deux_points_au_meme_instant() -> None:
    troncons = [(10.0, 0.01), (5.0, 0.005)]
    assert terme_instationnaire(0.0, troncons) == 0.0
    permanent = pression_en_aval_instationnaire(
        2.0e5,
        RHO,
        vitesse_amont=1.0,
        vitesse_aval=3.0,
        altitude_amont=0.0,
        altitude_aval=1.0,
        terme=0.0,
    )
    attendue = 2.0e5 + RHO * (1.0 - 9.0) / 2 + RHO * PESANTEUR * (0.0 - 1.0)
    assert proches(permanent, attendue, 1e-6)
    terme = terme_instationnaire(0.002, troncons)
    assert terme > 0
    variable = pression_en_aval_instationnaire(
        2.0e5,
        RHO,
        vitesse_amont=1.0,
        vitesse_aval=3.0,
        altitude_amont=0.0,
        altitude_aval=1.0,
        terme=terme,
    )
    assert proches(permanent - variable, RHO * terme, 1e-6)


def test_une_perte_de_charge_est_positive_et_une_perte_negative_est_refusee() -> None:
    assert perte_de_charge_reguliere(0.018, 10.0, 0.15, 0.6, 1000.0) > 0
    with pytest.raises(ValueError, match="positive"):
        pression_dans_l_aspiration(1.0e5, RHO, 2.0, 3.0, -1.0)


def test_une_surface_libre_large_est_pratiquement_au_repos() -> None:
    """Continuite : `V_surface / V_conduite = S_conduite / S_surface`."""
    section_conduite, section_surface, debit = 0.0177, 50.0, 10.6e-3
    v_surface = vitesse_dans_la_section(debit, section_surface)
    v_conduite = vitesse_dans_la_section(debit, section_conduite)
    assert proches_relativement(
        v_surface / v_conduite, section_conduite / section_surface, 1e-12
    )
    assert v_surface / v_conduite < 1e-3


def test_la_balle_alveolee_entre_en_crise_avant_la_sphere_lisse() -> None:
    assert REYNOLDS_CRITIQUE_BALLE_DE_GOLF < REYNOLDS_CRITIQUE_SPHERE_LISSE
    re_vol = 1.0e5
    assert est_apres_la_crise_de_trainee(re_vol, REYNOLDS_CRITIQUE_BALLE_DE_GOLF)
    assert not est_apres_la_crise_de_trainee(re_vol, REYNOLDS_CRITIQUE_SPHERE_LISSE)


def test_le_cd_de_la_sphere_est_presque_constant_puis_chute_a_la_crise() -> None:
    plateau = [coefficient_de_trainee_de_la_sphere(r) for r in (1e3, 1e4, 1e5, 2e5)]
    assert max(plateau) / min(plateau) < 1.3
    apres = coefficient_de_trainee_de_la_sphere(4.0e5)
    assert min(plateau) / apres > 2.0


def test_aux_grands_reynolds_le_frottement_ne_depend_que_de_la_rugosite() -> None:
    relative = 0.01
    limite = coefficient_en_zone_rugueuse(relative)
    for reynolds in (1.0e7, 1.0e8):
        assert proches_relativement(
            coefficient_de_colebrook(reynolds, relative), limite, 0.02
        )
    assert est_completement_rugueux(1.0e7, relative)
    assert not est_completement_rugueux(9.0e4, 1.0e-6)


def test_boussinesq_modele_le_tenseur_de_reynolds_sur_la_contrainte_visqueuse() -> None:
    mu_t, rho, k = 0.3, 1.2, 0.5
    gradient = ((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    r = tenseur_de_boussinesq(mu_t, gradient, rho, k)
    _, y, _ = ESPACE
    visqueuse = contraintes_visqueuses((y, 0, 0), mu_t)
    assert proches(r[0][1], float(visqueuse[0, 1]), 1e-12)
    assert proches(sum(r[i][i] for i in range(3)), -2 * rho * k, 1e-12)


def test_la_turbulence_aplatit_le_profil_et_creuse_la_paroi() -> None:
    proche_de_la_paroi = 0.01
    laminaire = 1 - (1 - proche_de_la_paroi) ** 2
    turbulent = vitesse_en_loi_de_puissance(proche_de_la_paroi, 1.0, 1.0)
    assert turbulent > 10 * laminaire


def test_les_coefficients_se_rapportent_a_leur_surface_de_reference() -> None:
    ecoulement = Ecoulement(MasseVolumique(1.225), 12.0)
    force = 3.0e3
    dessus = SurfacePortante(ecoulement, 30.0).coefficient(force)
    face = SurfacePortante(ecoulement, 3.0).coefficient(force)
    assert proches_relativement(face / dessus, 10.0, 1e-12)


def test_reynolds_et_froude_sont_des_rapports_de_forces() -> None:
    rho, v, longueur, mu = 1000.0, 2.0, 0.5, 1e-3
    inertie = rho * v**2 * longueur**2
    assert proches_relativement(
        inertie / (mu * v * longueur), nombre_de_reynolds(v, longueur, mu / rho), 1e-12
    )
    pesanteur = rho * PESANTEUR * longueur**3
    assert proches_relativement(
        inertie / pesanteur, nombre_de_froude(v, longueur) ** 2, 1e-12
    )
