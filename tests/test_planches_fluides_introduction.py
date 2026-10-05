"""Planches de mecanique des fluides, chapitre I : introduction.

Sources : `MecaniqueFluides/SLIDES/Slides_01_Introduction_2025_2026.pdf`
(51 planches) et `Slides_01_Introduction.pdf` (54 planches, l'annee
precedente), lues sur les pages rendues.

Les oracles du cours : le litre d'air de la planche 23, les deux courbes
de viscosite en fonction de la temperature (planche 33 et suivante), les
seuils 2 100 et 4 000 de l'experience de Reynolds, et l'exercice du
viscosimetre de l'ancien fascicule, dont la reponse imprimee est
`0,158 Pa.s`.
"""

from __future__ import annotations

import math

import pytest

from polytech.chemistry.domain.gaz import (
    AIR_SEC,
    masse_volumique_du_gaz_parfait,
    quantite_de_gaz,
)
from polytech.chemistry.domain.matiere import nombre_d_entites
from polytech.fluides.domain.cisaillement import (
    contrainte_tangentielle,
    force_de_cisaillement,
    viscosite_au_viscosimetre,
    vitesse_de_couette,
)
from polytech.fluides.domain.couche_limite import (
    epaisseur_de_couche_limite,
    reynolds_local,
)
from polytech.fluides.domain.milieu_continu import (
    est_rarefie,
    nombre_de_knudsen,
)
from polytech.fluides.domain.proprietes import viscosite_cinematique
from polytech.fluides.domain.regime_de_reynolds import (
    RegimeDeReynolds,
    nombre_de_reynolds,
    regime_en_conduite,
)
from polytech.fluides.domain.turbulence import Decomposition
from polytech.fluides.domain.viscosite import (
    contrainte_de_newton,
    viscosite_depuis_la_contrainte,
)
from polytech.fluides.domain.viscosite_et_temperature import (
    viscosite_d_un_gaz,
    viscosite_d_un_liquide,
)
from polytech.maths.domain.derivation_numerique import derivee_centree

ATMOSPHERE = 101_325.0
MESURE = {
    "couple": 1.8,
    "rayon": 0.06,
    "entrefer": 0.15e-2,
    "longueur": 0.40,
    "tours_par_minute": 300.0,
}
"""L'enonce du viscosimetre, en unites SI."""
AMBIANTE = 293.15


def test_un_litre_d_air_compte_quelque_dix_puissance_vingt_trois_molecules() -> None:
    """Planche 23 : « 1 litre d'air, 10^23 molecules » -- l'ordre de grandeur.

    La loi des gaz parfaits en donne `2,5 . 10^22` a vingt degres.
    """
    moles = quantite_de_gaz(ATMOSPHERE, 1e-3, AMBIANTE)
    molecules = nombre_d_entites(moles)
    assert molecules == pytest.approx(2.5e22, rel=0.01)
    assert round(math.log10(molecules)) == 22


def test_un_volume_ou_une_temperature_non_positifs_sont_refuses() -> None:
    with pytest.raises(ValueError, match="temperature non strictement positive"):
        quantite_de_gaz(ATMOSPHERE, 1e-3, 0.0)
    with pytest.raises(ValueError, match="volume negatif"):
        quantite_de_gaz(ATMOSPHERE, -1e-3, AMBIANTE)


def test_referencer_chaque_molecule_prendrait_des_centaines_de_milliers_d_annees() -> (
    None
):
    """Planche 23 : `10^10` operations par seconde, « 10^13 s ou 100.000 annees ».

    `10^13 s` font 317 000 ans : la planche arrondit a l'ordre de grandeur.
    """
    secondes = 1e23 / 1e10
    assert secondes == pytest.approx(1e13)
    annees = secondes / (365.25 * 24 * 3600)
    assert annees == pytest.approx(317_000, rel=0.01)


def test_le_nombre_de_knudsen_separe_le_continu_du_rarefie() -> None:
    """Planche 24 : `Kn = l / L`, et `Kn >= 1` pour un gaz rarefie."""
    assert nombre_de_knudsen(6.8e-8, 1e-3) == pytest.approx(6.8e-5)
    assert not est_rarefie(6.8e-5)
    assert est_rarefie(1.0)
    assert est_rarefie(nombre_de_knudsen(0.1, 0.05))


def test_une_echelle_nulle_n_a_pas_de_nombre_de_knudsen() -> None:
    with pytest.raises(ValueError, match="echelle nulle ou negative"):
        nombre_de_knudsen(6.8e-8, 0.0)
    with pytest.raises(ValueError, match="libre parcours negatif"):
        nombre_de_knudsen(-1.0, 1e-3)


def test_le_profil_de_couette_et_la_contrainte_de_cisaillement() -> None:
    """« Relation pour la viscosite » : `u(y) = y V / l`, `tau = F / A`."""
    assert vitesse_de_couette(0.0, 2e-3, 3.0) == 0
    assert vitesse_de_couette(2e-3, 2e-3, 3.0) == pytest.approx(3.0)
    assert vitesse_de_couette(1e-3, 2e-3, 3.0) == pytest.approx(1.5)
    assert contrainte_tangentielle(12.0, 0.5) == pytest.approx(24.0)
    gradient = 3.0 / 2e-3
    force = force_de_cisaillement(0.8, 0.5, gradient)
    assert contrainte_tangentielle(force, 0.5) == pytest.approx(
        contrainte_de_newton(0.8, gradient)
    )


def test_une_aire_ou_un_ecart_nuls_sont_refuses() -> None:
    with pytest.raises(ValueError, match="aire nulle ou negative"):
        contrainte_tangentielle(12.0, 0.0)
    with pytest.raises(ValueError, match="ecart nul ou negatif"):
        vitesse_de_couette(1e-3, 0.0, 3.0)
    with pytest.raises(ValueError, match="hors de l'entrefer"):
        vitesse_de_couette(3e-3, 2e-3, 3.0)


def test_le_viscosimetre_de_l_ancien_fascicule_donne_0_158_pascal_seconde() -> None:
    """« Viscosimetre » : 40 cm, diametre 12 cm, 0,15 cm, 300 tr/min, 1,8 N.m."""
    assert viscosite_au_viscosimetre(**MESURE) == pytest.approx(0.158, abs=5e-4)


def test_le_viscosimetre_n_est_que_la_loi_de_newton_sur_un_cylindre() -> None:
    """Le couple est la force `tau A` au rayon `R` ; le gradient vaut `omega R / l`."""
    rayon, entrefer, longueur = 0.06, 0.15e-2, 0.40
    vitesse = 300.0 * 2 * math.pi / 60 * rayon
    aire = 2 * math.pi * rayon * longueur
    contrainte = 1.8 / rayon / aire
    par_newton = viscosite_depuis_la_contrainte(contrainte, vitesse / entrefer)
    direct = viscosite_au_viscosimetre(
        couple=1.8,
        rayon=rayon,
        entrefer=entrefer,
        longueur=longueur,
        tours_par_minute=300.0,
    )
    assert direct == pytest.approx(par_newton)


def test_un_cylindre_immobile_ne_mesure_rien() -> None:
    with pytest.raises(ValueError, match="cylindre immobile"):
        viscosite_au_viscosimetre(**{**MESURE, "tours_par_minute": 0.0})
    with pytest.raises(ValueError, match="geometrie invalide"):
        viscosite_au_viscosimetre(**{**MESURE, "rayon": 0.0})


def test_la_viscosite_cinematique_de_l_air_suit_la_loi_des_gaz_parfaits() -> None:
    """Planche de la pancarte : `1,52 . 10^-5` a 1 atm, `0,380 . 10^-5` a 4 atm.

    Un lien avec la chimie : la masse volumique vient de `rho = P M / (R T)`.
    """
    for atmospheres, lue in ((1, 1.52e-5), (4, 0.380e-5)):
        rho = masse_volumique_du_gaz_parfait(
            atmospheres * ATMOSPHERE, AMBIANTE, AIR_SEC
        )
        assert viscosite_cinematique(1.83e-5, rho) == pytest.approx(lue, rel=5e-3)


SUTHERLAND = (1.458e-6, 110.4)
"""`a` et `b` de l'air, dans le livre de reference que nomme la planche 19."""

ANDRADE = (2.414e-5, 247.8, 140.0)
"""`a`, `b` et `c` de l'eau, meme source."""


def test_la_viscosite_d_un_gaz_croit_avec_la_temperature() -> None:
    """`mu = a T^(1/2) / (1 + b / T)` : la courbe va de 1,72 a 2,17 . 10^-5 Pa.s."""
    a_zero = viscosite_d_un_gaz(273.15, *SUTHERLAND)
    a_cent = viscosite_d_un_gaz(373.15, *SUTHERLAND)
    assert a_zero == pytest.approx(1.72e-5, rel=5e-3)
    assert a_cent == pytest.approx(2.17e-5, rel=5e-3)
    # La valeur de la pancarte, a vingt degres.
    assert viscosite_d_un_gaz(AMBIANTE, *SUTHERLAND) == pytest.approx(1.83e-5, rel=0.01)


def test_la_viscosite_d_un_liquide_decroit_avec_la_temperature() -> None:
    """`mu = a 10^(b / (T - c))` : la courbe va de 1,8 a 0,28 . 10^-3 Pa.s."""
    a_zero = viscosite_d_un_liquide(273.15, *ANDRADE)
    a_cent = viscosite_d_un_liquide(373.15, *ANDRADE)
    assert a_zero == pytest.approx(1.8e-3, rel=0.03)
    assert a_cent == pytest.approx(0.28e-3, rel=0.02)
    assert viscosite_d_un_liquide(293.15, *ANDRADE) == pytest.approx(1.0e-3, rel=0.01)


def test_les_gardes_des_lois_de_viscosite() -> None:
    with pytest.raises(ValueError, match="temperature absolue nulle ou negative"):
        viscosite_d_un_gaz(0.0, *SUTHERLAND)
    with pytest.raises(ValueError, match="temperature au plus egale a c"):
        viscosite_d_un_liquide(140.0, *ANDRADE)


def test_la_couche_limite_s_epaissit_comme_la_racine_de_l_abscisse() -> None:
    """« Comportement de la couche limite » : `delta(x) ~ racine(x)`."""
    proche = epaisseur_de_couche_limite(2.0, 0.1, 1.5e-5)
    loin = epaisseur_de_couche_limite(2.0, 0.4, 1.5e-5)
    assert loin / proche == pytest.approx(2.0)


def test_les_trois_regimes_de_l_experience_de_reynolds() -> None:
    """« Re < 2100 laminaire, Re > 4000 turbulent, 2100 < Re < 4000 transition »."""
    assert regime_en_conduite(1500.0) is RegimeDeReynolds.LAMINAIRE
    assert regime_en_conduite(3000.0) is RegimeDeReynolds.TRANSITION
    assert regime_en_conduite(2100.0) is RegimeDeReynolds.TRANSITION
    assert regime_en_conduite(4000.0) is RegimeDeReynolds.TRANSITION
    assert regime_en_conduite(5000.0) is RegimeDeReynolds.TURBULENT


def test_le_nombre_de_reynolds_de_la_cascade_de_kolmogorov() -> None:
    """`Re = U H / nu` : forces d'inertie sur forces de viscosite."""
    assert nombre_de_reynolds(2.0, 0.1, 1e-5) == pytest.approx(2e4)
    assert nombre_de_reynolds(2.0, 0.1, 1e-5) == reynolds_local(2.0, 0.1, 1e-5)


def test_les_gardes_du_nombre_de_reynolds() -> None:
    with pytest.raises(ValueError, match="viscosite cinematique nulle ou negative"):
        nombre_de_reynolds(2.0, 0.1, 0.0)
    with pytest.raises(ValueError, match="nombre de Reynolds negatif"):
        regime_en_conduite(-1.0)


def test_la_vitesse_turbulente_est_une_moyenne_plus_une_fluctuation() -> None:
    """« Ecoulement laminaire - turbulent » : `u(t) = u_barre + u'`, et `u_rms`."""
    laminaire = Decomposition((2.0, 2.0, 2.0, 2.0))
    assert laminaire.ecart_type() == 0
    turbulent = Decomposition((1.0, 3.0, 1.0, 3.0))
    assert turbulent.moyenne() == pytest.approx(2.0)
    assert turbulent.fluctuations() == pytest.approx((-1.0, 1.0, -1.0, 1.0))
    assert turbulent.ecart_type() == pytest.approx(1.0)


def test_la_difference_centree_est_exacte_sur_une_parabole() -> None:
    """Planche 15 : `df/dx ~ (f_(k+1) - f_(k-1)) / (2 h) + O(h^2)`."""

    def parabole(x: float) -> float:
        return 3 * x**2 - x + 2

    pas = 0.3
    approchee = derivee_centree(parabole(1.0 - pas), parabole(1.0 + pas), pas)
    assert approchee == pytest.approx(5.0)


def test_l_erreur_de_la_difference_centree_est_d_ordre_deux() -> None:
    """Diviser le pas par deux divise l'erreur par quatre."""
    erreurs = []
    for pas in (0.1, 0.05):
        approchee = derivee_centree(math.sin(1.0 - pas), math.sin(1.0 + pas), pas)
        erreurs.append(abs(approchee - math.cos(1.0)))
    assert erreurs[0] / erreurs[1] == pytest.approx(4.0, rel=0.01)


def test_un_pas_nul_n_a_pas_de_difference_centree() -> None:
    with pytest.raises(ValueError, match="pas nul ou negatif"):
        derivee_centree(1.0, 2.0, 0.0)
