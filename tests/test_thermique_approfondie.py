"""Cinq fascicules de thermique, repris en profondeur.

Ce que les planches enseignent et que les modules ne portaient pas.

**Convection naturelle** : le coefficient global de transmission `K`, et
les resistances de paroi plane, cylindrique et quelconque qui le
composent. C'est lui que les trois applications du fascicule emploient --
la tablette, le double vitrage, le boiler -- et c'est lui qui repond a sa
question : « comment reduire les pertes au travers des vitrages ».

**Convection forcee** : l'**analyse dimensionnelle** qui fonde toutes les
correlations. Le fascicule y consacre six planches -- denombrement des
variables, choix des variables, etapes de la demarche -- parce que c'est
elle qui explique pourquoi `Nu = f(Re, Pr)` et pas autre chose.

**Rayonnement** : la loi de Planck, celle de Stefan-Boltzmann, et la loi
de **Kirchhoff** qui les relie aux corps reels -- l'emissivite egale
l'absorptivite a l'equilibre, et c'est ce qui rend l'effet de serre
calculable.

**Echanges entre surfaces** : les cas particuliers que le fascicule
tabule, dont les deux plans gris infinis paralleles, et la troisieme
surface adiabatique qui reradie tout ce qu'elle recoit.

**Echangeurs** : l'efficacite, les configurations d'ecoulement, et les
equations generales.

Le point fixe le plus net est celui de Kirchhoff : a l'equilibre
thermique, `epsilon = alpha` exactement, et un corps gris le verifie a
toute temperature.
"""

from __future__ import annotations

import math

import pytest

from polytech.thermique.domain.convection.forcee import (
    VARIABLES_DE_LA_CONVECTION_FORCEE,
    groupes_sans_dimension,
    nombre_de_groupes_de_vaschy_buckingham,
)
from polytech.thermique.domain.convection.naturelle import (
    coefficient_global_de_transmission,
    resistance_de_paroi_cylindrique,
    resistance_de_paroi_plane,
    resistance_de_surface,
)
from polytech.thermique.domain.echangeurs import (
    Echangeur,
    capacite_minimale,
    efficacite_a_contre_courant,
    efficacite_a_courants_paralleles,
    flux_maximal_echangeable,
)
from polytech.thermique.domain.rayonnement.corps_noir import (
    absorptivite_de_kirchhoff,
    fraction_emise_sous,
)
from polytech.thermique.domain.rayonnement.surfaces_grises import (
    flux_entre_deux_plans_gris,
    resistance_d_une_surface_adiabatique,
)


def test_la_resistance_d_une_paroi_plane_croit_avec_son_epaisseur() -> None:
    """`e / (lambda S)` : la forme la plus simple, et celle du vitrage."""
    mince = resistance_de_paroi_plane(0.004, 1.0, 1.0)
    epaisse = resistance_de_paroi_plane(0.008, 1.0, 1.0)
    assert epaisse == pytest.approx(2 * mince)


def test_la_resistance_d_une_paroi_cylindrique_est_logarithmique() -> None:
    """`ln(r_e/r_i) / (2 pi lambda L)` : ce n'est pas `e / (lambda S)`.

    Transposer le cas plan au tube est l'erreur du chapitre, et l'ecart
    n'est pas negligeable des que l'epaisseur n'est plus petite devant le
    rayon.
    """
    attendu = math.log(2.0) / (2 * math.pi * 1.0 * 1.0)
    assert resistance_de_paroi_cylindrique(0.01, 0.02, 1.0, 1.0) == pytest.approx(
        attendu,
    )


def test_une_paroi_cylindrique_inversee_est_refusee() -> None:
    with pytest.raises(ValueError, match="rayon exterieur"):
        resistance_de_paroi_cylindrique(0.02, 0.01, 1.0, 1.0)


def test_le_coefficient_global_est_l_inverse_de_la_somme_des_resistances() -> None:
    """`1 / (K S) = somme R` : c'est la mise en serie, et rien d'autre."""
    interieur = resistance_de_surface(8.0, 1.0)
    paroi = resistance_de_paroi_plane(0.004, 1.0, 1.0)
    exterieur = resistance_de_surface(25.0, 1.0)
    global_ = coefficient_global_de_transmission((interieur, paroi, exterieur), 1.0)
    assert global_ == pytest.approx(1 / (interieur + paroi + exterieur))


def test_le_double_vitrage_reduit_les_pertes_du_simple() -> None:
    """La question du fascicule, et sa reponse chiffree.

    Ajouter une lame d'air de seize millimetres entre deux vitres ajoute sa
    resistance en serie -- et comme l'air conduit cent fois moins que le
    verre, elle domine.
    """
    verre, air = 1.0, 0.025
    simple = coefficient_global_de_transmission(
        (
            resistance_de_surface(8.0, 1.0),
            resistance_de_paroi_plane(0.004, verre, 1.0),
            resistance_de_surface(25.0, 1.0),
        ),
        1.0,
    )
    double = coefficient_global_de_transmission(
        (
            resistance_de_surface(8.0, 1.0),
            resistance_de_paroi_plane(0.004, verre, 1.0),
            resistance_de_paroi_plane(0.016, air, 1.0),
            resistance_de_paroi_plane(0.004, verre, 1.0),
            resistance_de_surface(25.0, 1.0),
        ),
        1.0,
    )
    assert double < simple / 2


def test_une_liste_de_resistances_vide_est_refusee() -> None:
    with pytest.raises(ValueError, match="aucune resistance"):
        coefficient_global_de_transmission((), 1.0)


def test_le_theoreme_de_vaschy_buckingham_donne_le_nombre_de_groupes() -> None:
    """`n - k` : variables moins dimensions fondamentales.

    C'est l'etape que le fascicule detaille sur six planches, et elle
    explique pourquoi la convection forcee se correle par **deux** nombres
    sans dimension et non par sept variables.
    """
    assert nombre_de_groupes_de_vaschy_buckingham(variables=7, dimensions=4) == 3


def test_un_nombre_de_variables_inferieur_aux_dimensions_est_refuse() -> None:
    """Il n'y aurait aucun groupe a former, et la demarche n'a pas de sens."""
    with pytest.raises(ValueError, match="moins de variables"):
        nombre_de_groupes_de_vaschy_buckingham(variables=2, dimensions=4)


def test_les_variables_de_la_convection_forcee_sont_celles_du_fascicule() -> None:
    """Sept variables, quatre dimensions, donc trois groupes : Nu, Re, Pr."""
    attendu = 7
    assert len(VARIABLES_DE_LA_CONVECTION_FORCEE) == attendu
    assert groupes_sans_dimension() == ("Nusselt", "Reynolds", "Prandtl")


def test_la_loi_de_kirchhoff_egale_emissivite_et_absorptivite() -> None:
    """A l'equilibre thermique, exactement. C'est ce qui ferme le chapitre.

    Sans elle, un corps gris aurait deux proprietes independantes et les
    bilans radiatifs ne se fermeraient pas.
    """
    assert absorptivite_de_kirchhoff(0.85) == pytest.approx(0.85)


def test_une_emissivite_hors_de_zero_un_est_refusee_par_kirchhoff() -> None:
    with pytest.raises(ValueError, match="emissivite hors de"):
        absorptivite_de_kirchhoff(1.5)


def test_la_fraction_emise_sous_une_longueur_d_onde_croit_avec_elle() -> None:
    """La « loi integree » que le fascicule donne en abaque.

    A `lambda T` tres petit la fraction tend vers zero, a `lambda T` tres
    grand elle tend vers un : ce sont les deux bornes exactes, et elles
    encadrent toute lecture d'abaque.
    """
    assert fraction_emise_sous(1e-7, 300.0) == pytest.approx(0.0, abs=1e-6)
    assert fraction_emise_sous(1e-2, 300.0) == pytest.approx(1.0, abs=1e-3)
    milieu = fraction_emise_sous(1e-5, 300.0)
    assert 0.0 < milieu < 1.0


def test_le_spectre_solaire_est_essentiellement_visible_et_proche_infrarouge() -> None:
    """Application « effet de serre » : a 5800 K, presque tout passe sous 3 um.

    C'est le fondement de l'effet de serre -- le verre est transparent a
    ce rayonnement et opaque a celui, bien plus long, que la piece
    reemet.
    """
    assert fraction_emise_sous(3e-6, 5800.0) > 0.95


def test_deux_plans_gris_paralleles_echangent_selon_la_formule_tabulee() -> None:
    """`sigma (T1^4 - T2^4) / (1/eps_1 + 1/eps_2 - 1)`.

    Cas particulier que le fascicule tabule, et qui sert partout -- deux
    vitres, deux parois d'un four, un ecran anti-rayonnement.
    """
    flux = flux_entre_deux_plans_gris(0.8, 0.8, 500.0, 300.0, aire=1.0)
    denominateur = 1 / 0.8 + 1 / 0.8 - 1
    attendu = 5.670374419e-8 * (500.0**4 - 300.0**4) / denominateur
    assert flux == pytest.approx(attendu)


def test_deux_plans_noirs_echangent_la_difference_des_emittances() -> None:
    """Cas limite : les deux emissivites a un rendent le denominateur unite."""
    flux = flux_entre_deux_plans_gris(1.0, 1.0, 500.0, 300.0, aire=1.0)
    attendu = 5.670374419e-8 * (500.0**4 - 300.0**4)
    assert flux == pytest.approx(attendu)


def test_un_ecran_reduit_l_echange_entre_deux_plans() -> None:
    """Propriete physique : baisser une emissivite reduit l'echange."""
    fort = flux_entre_deux_plans_gris(0.9, 0.9, 500.0, 300.0, aire=1.0)
    faible = flux_entre_deux_plans_gris(0.1, 0.1, 500.0, 300.0, aire=1.0)
    assert faible < fort / 10


def test_une_surface_adiabatique_n_oppose_aucune_resistance_nette() -> None:
    """Elle reradie tout ce qu'elle recoit : son flux net est nul.

    Le fascicule lui consacre une planche -- « 3 surfaces dont 1 surface
    adiabatique » -- parce qu'elle change le schema sans ajouter
    d'inconnue de temperature.
    """
    assert resistance_d_une_surface_adiabatique() == 0.0


def test_la_capacite_minimale_est_celle_qui_limite_l_echange() -> None:
    """`(q c)_min` : c'est elle qui fixe le flux maximal echangeable."""
    assert capacite_minimale(1000.0, 2000.0) == pytest.approx(1000.0)


def test_le_flux_maximal_echangeable_est_celui_du_fluide_limitant() -> None:
    """`(q c)_min (T_chaud_entree - T_froid_entree)` : la borne absolue."""
    maximal = flux_maximal_echangeable(1000.0, 2000.0, 80.0, 20.0)
    assert maximal == pytest.approx(1000.0 * 60.0)


def test_un_ecart_d_entree_inverse_est_refuse() -> None:
    """Le fluide « chaud » serait le plus froid : l'echangeur fonctionne a rebours."""
    with pytest.raises(ValueError, match="fluide chaud"):
        flux_maximal_echangeable(1000.0, 2000.0, 20.0, 80.0)


def test_le_contre_courant_est_plus_efficace_que_les_courants_paralleles() -> None:
    """C'est le resultat que le fascicule met en avant, a NUT et rapport egaux.

    Et il est exact : a contre-courant l'ecart de temperature reste a peu
    pres constant le long de l'echangeur, au lieu de s'effondrer.
    """
    nut, rapport = 1.5, 0.5
    assert efficacite_a_contre_courant(nut, rapport) > (
        efficacite_a_courants_paralleles(nut, rapport)
    )


def test_les_deux_efficacites_restent_entre_zero_et_un() -> None:
    """Une efficacite est une fraction du flux maximal : elle ne le depasse pas."""
    for nut in (0.1, 1.0, 10.0):
        for rapport in (0.1, 0.5, 1.0):
            for efficacite in (
                efficacite_a_contre_courant(nut, rapport),
                efficacite_a_courants_paralleles(nut, rapport),
            ):
                assert 0.0 < efficacite < 1.0


def test_un_nut_nul_donne_une_efficacite_nulle_et_est_refuse() -> None:
    with pytest.raises(ValueError, match="NUT nul"):
        efficacite_a_contre_courant(0.0, 0.5)


def test_un_rapport_de_capacites_hors_de_l_intervalle_est_refuse() -> None:
    with pytest.raises(ValueError, match="rapport de capacites"):
        efficacite_a_courants_paralleles(1.0, 1.5)


def test_l_echangeur_du_tp_4_garde_ses_valeurs() -> None:
    """Non-regression : les ajouts ne touchent pas a l'ancrage du TP 4."""
    echangeur = Echangeur(conductance=775.61)
    assert echangeur.coefficient_rapporte(1.1556) == pytest.approx(671.17, abs=0.1)
