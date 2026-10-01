"""Les classifications d'ecoulement du chapitre I, et la cascade de Kolmogorov.

Meme chapitre que `test_proprietes_des_fluides`, et meme raison d'avoir
manque : le fascicule « Introduction » n'etait pas declare.

Deux contenus que la planche donne en clair et qui decident de tout le
reste du cours :

* le critere d'incompressibilite. « Les ecoulements gazeux consideres
  comme incompressibles si les variations de masse volumique < 5 %
  (Ma < 0,3) ». Les deux bornes vont ensemble, et le test verifie
  qu'elles sont bien coherentes -- ce qui n'est pas evident, l'une etant
  cinematique et l'autre thermodynamique ;
* la cascade de Kolmogorov, dont la planche donne l'exposant `-5/3` et le
  sens -- injection d'energie aux grandes echelles, dissipation visqueuse
  aux petites, transfert entre les deux.
"""

from __future__ import annotations

import pytest

from polytech.fluides.domain.regimes import (
    EXPOSANT_DE_KOLMOGOROV,
    MACH_INCOMPRESSIBLE,
    densite_spectrale_de_turbulence,
    est_incompressible,
    nombre_de_mach,
    variation_de_masse_volumique,
)

CELERITE_DU_SON_DANS_L_AIR = 343.0


def test_le_nombre_de_mach_rapporte_la_vitesse_a_celle_du_son() -> None:
    """`Ma = V / c`, et la planche l'ecrit en ces termes."""
    assert nombre_de_mach(103.0, CELERITE_DU_SON_DANS_L_AIR) == pytest.approx(
        0.3, abs=0.01
    )


def test_le_seuil_d_incompressibilite_est_a_mach_zero_virgule_trois() -> None:
    """Soit 103 m/s dans l'air, c'est-a-dire 370 km/h.

    L'ordre de grandeur est ce qui rend le critere utile : tout
    ecoulement d'air de genie civil, de ventilation ou d'automobile
    courante est donc incompressible, et c'est pourquoi le cours peut
    traiter l'incompressible d'abord.
    """
    assert est_incompressible(nombre_de_mach(100.0, CELERITE_DU_SON_DANS_L_AIR))
    assert not est_incompressible(nombre_de_mach(200.0, CELERITE_DU_SON_DANS_L_AIR))
    assert pytest.approx(0.3) == MACH_INCOMPRESSIBLE


def test_les_deux_bornes_du_critere_sont_coherentes() -> None:
    """« variations de masse volumique < 5 % » et « Ma < 0,3 » s'accordent.

    La planche donne les deux comme si elles allaient de soi ensemble, et
    ce n'est pas evident : l'une est cinematique, l'autre
    thermodynamique. La relation isentropique les relie, et a `Ma = 0,3`
    la variation relative de masse volumique vaut environ 4,4 % -- juste
    sous les cinq pour cent annonces. Les deux bornes disent donc bien la
    meme chose, et le critere n'est pas arbitraire.
    """
    variation = variation_de_masse_volumique(MACH_INCOMPRESSIBLE)
    assert variation == pytest.approx(0.044, abs=0.004)
    assert variation < 0.05


def test_la_variation_de_masse_volumique_croit_avec_le_mach() -> None:
    """Et elle croit comme le **carre** du Mach aux petites vitesses.

    C'est ce qui rend le seuil si permissif : doubler la vitesse
    quadruple l'erreur commise en negligeant la compressibilite, de sorte
    qu'a Mach 0,15 elle n'est plus que de un pour cent.
    """
    petite = variation_de_masse_volumique(0.15)
    grande = variation_de_masse_volumique(0.30)
    assert grande / petite == pytest.approx(4.0, rel=0.05)


def test_la_cascade_de_kolmogorov_decroit_en_puissance_moins_cinq_tiers() -> None:
    """`E(k) ~ k^(-5/3)`, exposant que la planche donne explicitement.

    Il dit que l'energie se repartit sur toutes les echelles entre
    l'injection et la dissipation, et qu'elle decroit vers les petites.
    C'est pourquoi une simulation directe de turbulence coute si cher :
    il faut resoudre toutes les echelles jusqu'a la dissipation.
    """
    assert pytest.approx(-5 / 3) == EXPOSANT_DE_KOLMOGOROV
    une = densite_spectrale_de_turbulence(nombre_d_onde=1.0, constante=2.0)
    dix = densite_spectrale_de_turbulence(nombre_d_onde=10.0, constante=2.0)
    assert une / dix == pytest.approx(10 ** (5 / 3))


def test_une_celerite_du_son_nulle_est_refusee() -> None:
    """Le nombre de Mach y diverge."""
    with pytest.raises(ValueError, match="celerite du son nulle ou negative"):
        nombre_de_mach(100.0, 0.0)


def test_un_nombre_d_onde_nul_est_refuse() -> None:
    """La densite spectrale y diverge, et la cascade ne commence pas la.

    Le nombre d'onde nul correspondrait a une echelle infinie ; la
    cascade commence a l'echelle d'injection, qui est finie.
    """
    with pytest.raises(ValueError, match="nombre d'onde nul ou negatif"):
        densite_spectrale_de_turbulence(nombre_d_onde=0.0, constante=1.0)


def test_un_mach_negatif_est_refuse() -> None:
    """Le sens de l'ecoulement ne change pas sa compressibilite."""
    with pytest.raises(ValueError, match="nombre de Mach negatif"):
        variation_de_masse_volumique(-0.2)
