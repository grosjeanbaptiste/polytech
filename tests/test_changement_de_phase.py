"""Les chapitres 5 et 6 du tome 2 : condensation et ebullition.

Absents de la bibliotheque pour la meme raison que les precedents -- la
mesure ne regardait que les planches.

Le syllabus donne la prose et les ordres de grandeur en clair, et ses
formules en images. Ce qui est teste ici est donc ce que la prose
etablit, et surtout les **nombres qu'elle annonce**, qui sont ce qu'on
peut verifier sans voir les formules :

* la vapeur acquiert une vitesse normale a la paroi de l'ordre de
  `0,1 m/s` -- le syllabus le chiffre, et le bilan de masse le redonne ;
* cette vitesse « correspond a une energie cinetique tres faible, et en
  appliquant le theoreme de Bernoulli, a une chute de pression (donc de
  temperature de saturation) negligeable » -- l'argument est refait ici,
  et il est ecrasant : trois millipascals contre un bar ;
* en refroidissement VLSI « des densites de flux de 100000 W/m^2 sont
  depassees » -- c'est ce qui justifie le chapitre.
"""

from __future__ import annotations

import pytest

from polytech.thermique.domain.condensation import (
    ACTIONS_DE_LA_CONDENSATION,
    chute_de_pression_dynamique,
    vitesse_normale_de_la_vapeur,
)
from polytech.thermique.domain.ebullition import (
    FLUX_DES_CIRCUITS_DENSES,
    REGIMES_EN_CONVECTION_NATURELLE,
    Pilotage,
    risque_de_burn_out,
)

# Vapeur d'eau saturee a un bar, valeurs de la litterature.
MASSE_VOLUMIQUE_VAPEUR = 0.59
CHALEUR_LATENTE = 2.257e6
PRESSION_ATMOSPHERIQUE = 1.013e5


def test_la_condensation_se_decompose_en_trois_actions_successives() -> None:
    """Le syllabus les numerote, et l'ordre porte le raisonnement.

    Il faut d'abord les enumerer pour pouvoir comparer « les
    contributions respectives de ces trois phenomenes a la resistance de
    chaleur » -- ce que le chapitre fait ensuite pour montrer que seule
    la troisieme compte, pour une vapeur pure.
    """
    assert len(ACTIONS_DE_LA_CONDENSATION) == 3
    assert ACTIONS_DE_LA_CONDENSATION[0].startswith("deplacement")
    assert ACTIONS_DE_LA_CONDENSATION[-1].startswith("transmission")


def test_la_vitesse_normale_de_la_vapeur_est_du_dixieme_de_metre_par_seconde() -> None:
    """« u_v ~= 0,1 m/s », chiffre par le syllabus pour la vapeur d'eau.

    Le bilan le redonne : le flux `phi` condense `phi / l_v` kilogrammes
    par seconde et par metre carre, que la vapeur doit amener a la
    vitesse `phi / (rho_v l_v)`. Avec un flux de condenseur de l'ordre de
    `1,3 . 10^5 W/m^2`, on retombe bien sur le dixieme de metre par
    seconde -- ce qui valide a la fois l'expression et l'ordre de
    grandeur annonce.
    """
    vitesse = vitesse_normale_de_la_vapeur(
        flux=1.33e5,
        masse_volumique=MASSE_VOLUMIQUE_VAPEUR,
        chaleur_latente=CHALEUR_LATENTE,
    )
    assert vitesse == pytest.approx(0.1, rel=0.05)


def test_la_vitesse_croit_avec_le_flux_et_decroit_avec_la_chaleur_latente() -> None:
    """Deux fois plus de flux, deux fois plus vite ; et l'inverse pour `l_v`.

    La chaleur latente au denominateur est ce qui rend la condensation
    si efficace : une vapeur a forte chaleur latente transporte beaucoup
    d'energie pour peu de masse, donc pour peu de vitesse.
    """
    reference = vitesse_normale_de_la_vapeur(
        1e5, MASSE_VOLUMIQUE_VAPEUR, CHALEUR_LATENTE
    )
    double = vitesse_normale_de_la_vapeur(2e5, MASSE_VOLUMIQUE_VAPEUR, CHALEUR_LATENTE)
    latente = vitesse_normale_de_la_vapeur(
        1e5,
        MASSE_VOLUMIQUE_VAPEUR,
        2 * CHALEUR_LATENTE,
    )
    assert double == pytest.approx(2 * reference)
    assert latente == pytest.approx(reference / 2)


def test_la_chute_de_pression_de_la_vapeur_est_negligeable() -> None:
    """L'argument de Bernoulli du syllabus, refait et chiffre.

    `rho u^2 / 2` sur une vapeur a `0,59 kg/m^3` lancee a `0,1 m/s` vaut
    trois millipascals. Rapporte a la pression atmospherique, c'est sept
    ordres de grandeur en dessous : la temperature de saturation n'en
    bouge pas, et c'est ce qui autorise a ne garder que la resistance du
    film de condensat.

    Le syllabus precise que « ce raisonnement n'est correct que si la
    vapeur est pure » : avec des incondensables, la vapeur doit diffuser
    et le potentiel thermique chute pour de bon.
    """
    chute = chute_de_pression_dynamique(MASSE_VOLUMIQUE_VAPEUR, 0.1)
    assert chute == pytest.approx(0.00295, rel=0.01)
    assert chute / PRESSION_ATMOSPHERIQUE < 1e-6


def test_une_masse_volumique_nulle_est_refusee() -> None:
    """La vitesse serait infinie : une vapeur sans masse n'existe pas."""
    with pytest.raises(ValueError, match="masse volumique nulle ou negative"):
        vitesse_normale_de_la_vapeur(1e5, 0.0, CHALEUR_LATENTE)


def test_une_chaleur_latente_nulle_est_refusee() -> None:
    """Sans chaleur latente il n'y a pas de changement d'etat a transmettre."""
    with pytest.raises(ValueError, match="chaleur latente nulle ou negative"):
        vitesse_normale_de_la_vapeur(1e5, MASSE_VOLUMIQUE_VAPEUR, 0.0)


def test_les_regimes_d_ebullition_sont_enumeres_dans_l_ordre() -> None:
    """L'ordre **est** le contenu : c'est la courbe de Nukiyama du chapitre.

    Convection naturelle, puis ebullition nucleee, puis la crise, puis
    l'ebullition en film. Les intervertir perdrait le fait que la crise
    separe deux regimes et non deux etats.
    """
    assert REGIMES_EN_CONVECTION_NATURELLE[0] == "convection naturelle"
    assert REGIMES_EN_CONVECTION_NATURELLE[1] == "ebullition nucleee"
    assert REGIMES_EN_CONVECTION_NATURELLE[-1] == "ebullition en film"
    assert "crise" in REGIMES_EN_CONVECTION_NATURELLE[2]


def test_le_flux_des_circuits_denses_depasse_cent_kilowatts_par_metre_carre() -> None:
    """« des densites de flux de 100000 W/m^2 sont depassees », dit le syllabus.

    C'est le chiffre qui justifie le chapitre : a ce niveau, ni la
    convection naturelle ni la convection forcee en gaz ne suffisent, et
    le changement de phase devient le seul recours.
    """
    assert pytest.approx(1e5) == FLUX_DES_CIRCUITS_DENSES


def test_un_flux_impose_au_dela_du_critique_fait_courir_le_burn_out() -> None:
    """Et c'est le pilotage, non le flux seul, qui decide du danger.

    A flux impose -- un element electrique, un crayon de reacteur -- la
    paroi doit evacuer ce qu'on lui donne ; passe le flux critique elle
    ne peut plus qu'en montant brutalement en temperature, d'ou le nom de
    « burn-out ». A temperature imposee -- un condenseur, un bain --, le
    meme depassement fait seulement chuter le flux, sans destruction.

    C'est la distinction que le chapitre porte, et la raison pour
    laquelle la meme courbe se lit de deux facons.
    """
    assert risque_de_burn_out(Pilotage.FLUX_IMPOSE, flux=1.2e6, critique=1e6)
    assert not risque_de_burn_out(Pilotage.FLUX_IMPOSE, flux=0.8e6, critique=1e6)
    assert not risque_de_burn_out(
        Pilotage.TEMPERATURE_IMPOSEE, flux=1.2e6, critique=1e6
    )


def test_un_flux_critique_nul_est_refuse() -> None:
    """Une surface qui ne tiendrait aucun flux n'est pas un cas du cours."""
    with pytest.raises(ValueError, match="flux critique nul ou negatif"):
        risque_de_burn_out(Pilotage.FLUX_IMPOSE, flux=1e5, critique=0.0)
