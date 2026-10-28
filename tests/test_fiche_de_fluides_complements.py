"""Ce que la fiche de fluides affirmait sans qu'aucune fonction le porte.

Loi `1/7`, crise de trainee de la balle de golf, zone rugueuse, point
d'aspiration d'une pompe, Bernoulli entre deux points : chaque affirmation
de `cursus/fiches/MecaniqueFluides.toml` est relue dans l'article, puis
calculee par `polytech.fluides`.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from polytech.fluides.domain.boussinesq import flottabilite_volumique
from polytech.fluides.domain.cavitation import (
    PRESSION_DE_VAPEUR_EAU_20C,
    npsh_disponible,
    pression_dans_l_aspiration,
    risque_de_cavitation,
)
from polytech.fluides.domain.conduite import coefficient_de_blasius
from polytech.fluides.domain.fondamentales.bernoulli_instationnaire import (
    pression_en_aval_instationnaire,
    terme_instationnaire,
)
from polytech.fluides.domain.loi_de_puissance import (
    EXPOSANT_DE_BLASIUS,
    rapport_vitesse_moyenne_sur_maximale,
    vitesse_en_loi_de_puissance,
)
from polytech.fluides.domain.pertes_de_charge import coefficient_de_colebrook
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

FICHE = Path(__file__).resolve().parent.parent / "cursus/fiches/MecaniqueFluides.toml"


def _article(debut: str) -> dict[str, Any]:
    articles = tomllib.loads(FICHE.read_text(encoding="utf-8"))["article"]
    return next(a for a in articles if a["notion"].startswith(debut))


def test_le_profil_turbulent_de_la_fiche_est_la_loi_un_septieme() -> None:
    assert f"^(1/{EXPOSANT_DE_BLASIUS})" in _article("regimes_laminaire")["figure"]
    assert vitesse_en_loi_de_puissance(0.5, 1.0, 1.0) > 1 - 0.5**2
    assert rapport_vitesse_moyenne_sur_maximale() == pytest.approx(49 / 60)


def test_la_loi_un_septieme_accompagne_blasius_aux_grands_reynolds() -> None:
    assert coefficient_de_blasius(9.0e4) == pytest.approx(0.0182, abs=5e-5)


def test_la_balle_de_golf_a_franchi_la_crise_la_ou_la_sphere_lisse_non() -> None:
    cas = _article("trainee_en_pratique")["cas"][0]["texte"]
    assert "balle de golf" in cas
    reynolds_du_jeu = 2.0e5
    assert est_apres_la_crise_de_trainee(
        reynolds_du_jeu, REYNOLDS_CRITIQUE_BALLE_DE_GOLF
    )
    assert not est_apres_la_crise_de_trainee(
        reynolds_du_jeu, REYNOLDS_CRITIQUE_SPHERE_LISSE
    )
    assert coefficient_de_trainee_de_la_sphere(
        REYNOLDS_CRITIQUE_SPHERE_LISSE
    ) < 0.5 * coefficient_de_trainee_de_la_sphere(1e5)


def test_aux_grands_reynolds_le_frottement_ne_depend_que_de_la_rugosite_relative() -> (
    None
):
    assert "rugosité relative" in _article("ecoulement_de_poiseuille")["raison"]
    assert est_completement_rugueux(1e7, 0.02)
    assert coefficient_de_colebrook(1e7, 0.02) == pytest.approx(
        coefficient_de_colebrook(1e9, 0.02), rel=1e-2
    )
    assert coefficient_de_colebrook(1e9, 0.02) == pytest.approx(
        coefficient_en_zone_rugueuse(0.02), rel=1e-2
    )


def test_le_point_d_aspiration_a_une_pression_inferieure_a_celle_de_la_surface() -> (
    None
):
    assert "aspiration" in _article("equation_de_bernoulli")["piege"]
    p = pression_dans_l_aspiration(101_325.0, 1000.0, 1.5, 0.6, 500.0)
    assert p < 101_325.0
    assert not risque_de_cavitation(p, PRESSION_DE_VAPEUR_EAU_20C)
    assert npsh_disponible(p, 0.6, PRESSION_DE_VAPEUR_EAU_20C, 1000.0) > 0


def test_une_aspiration_trop_haute_cavite() -> None:
    p = pression_dans_l_aspiration(101_325.0, 1000.0, 10.5, 0.6, 500.0)
    assert risque_de_cavitation(p, PRESSION_DE_VAPEUR_EAU_20C)


def test_bernoulli_s_ecrit_a_un_instant_et_le_terme_instationnaire_le_dit() -> None:
    assert "deux instants" in _article("equation_de_bernoulli")["piege"]
    stationnaire = pression_en_aval_instationnaire(2e5, 1000.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    terme = terme_instationnaire(0.01, [(5.0, 0.01)])
    instationnaire = pression_en_aval_instationnaire(
        2e5, 1000.0, 1.0, 1.0, 0.0, 0.0, terme
    )
    assert stationnaire - instationnaire == pytest.approx(1000.0 * 5.0)


def test_un_fluide_plus_chaud_subit_une_flottabilite_vers_le_haut() -> None:
    assert flottabilite_volumique(1.2, 3.3e-3, 305.0, 300.0) > 0


def test_la_boussinesq_de_la_fiche_est_la_viscosite_turbulente() -> None:
    assert "Boussinesq" in _article("moyenne_de_reynolds")["raison"]
    gradient = ((0.0, 2.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert tenseur_de_boussinesq(0.5, gradient, 1.2, 0.0)[0][1] == pytest.approx(1.0)
