"""Le chapitre 4 du tome 1 du syllabus : le rayonnement des gaz.

Ce chapitre existait dans le cours et pas dans la bibliotheque, parce que
la mesure de couverture ne regardait pas les syllabus -- elle ne voyait
que les planches. Il porte l'absorption d'un gaz de combustion, et il
sert les chambres de combustion et les ecrans de tubes.

Le syllabus en donne la prose en clair ; ses formules sont des images que
la conversion n'a pas rendues. Ce qui est implemente est donc exactement
ce que la prose etablit -- la loi de Beer, l'opacite comme produit `p l`,
la proportionnalite du coefficient a la pression partielle -- et ce
qu'elle ne donne qu'en abaque est **refuse**, non interpole.
"""

from __future__ import annotations

import math

import pytest

from polytech.thermique.domain.rayonnement_des_gaz import (
    absorptivite_du_trajet,
    coefficient_lineaire,
    longueur_moyenne_de_rayonnement,
    opacite,
    transmittance,
)


def test_un_trajet_nul_laisse_passer_tout_le_rayonnement() -> None:
    """`exp(0) = 1` : sans gaz a traverser, rien n'est absorbe."""
    assert transmittance(coefficient=0.8, longueur=0.0) == pytest.approx(1.0)
    assert absorptivite_du_trajet(coefficient=0.8, longueur=0.0) == pytest.approx(0.0)


def test_la_transmittance_et_l_absorptivite_se_completent() -> None:
    """La loi de Beer ne laisse que deux issues : transmis ou absorbe.

    Aucune diffusion dans ce modele -- le syllabus traite un milieu
    diathermane absorbant, pas diffusant. La somme vaut donc un
    exactement, et c'est la propriete qui garde le calcul.
    """
    for longueur in (0.1, 1.0, 3.7):
        transmis = transmittance(coefficient=0.45, longueur=longueur)
        absorbe = absorptivite_du_trajet(coefficient=0.45, longueur=longueur)
        assert transmis + absorbe == pytest.approx(1.0)


def test_la_transmittance_decroit_exponentiellement_avec_le_trajet() -> None:
    """Doubler le trajet **eleve au carre** la transmittance, il ne la divise pas.

    C'est le point que la loi de Beer rend contre-intuitif, et l'erreur
    qu'un calcul lineaire ferait : `exp(-2al) = (exp(-al))^2`.
    """
    simple = transmittance(coefficient=0.3, longueur=2.0)
    double = transmittance(coefficient=0.3, longueur=4.0)
    assert double == pytest.approx(simple**2)


def test_l_opacite_est_le_produit_de_la_pression_partielle_et_du_trajet() -> None:
    """Le syllabus pose l'opacite comme le produit de ses deux facteurs.

    « Le produit (p l) est appele opacite du milieu diathermane », 4.2.1.
    L'opacite est donc une grandeur de **deux** facteurs : un gaz peu
    concentre sur un long trajet absorbe autant qu'un gaz concentre sur
    un trajet court.
    """
    assert opacite(pression_partielle=0.2, longueur=5.0) == pytest.approx(1.0)
    assert opacite(pression_partielle=1.0, longueur=1.0) == pytest.approx(1.0)


def test_le_coefficient_lineaire_est_proportionnel_a_la_pression_partielle() -> None:
    """Le coefficient suit la concentration, donc la pression partielle.

    Le syllabus le dit en ces termes, 4.2.1 : « proportionnel a la
    concentration en molecules, c'est-a-dire a la pression partielle p ».
    """
    simple = coefficient_lineaire(constante=2.5, pression_partielle=0.1)
    double = coefficient_lineaire(constante=2.5, pression_partielle=0.2)
    assert double == pytest.approx(2 * simple)


def test_deux_gaz_de_meme_opacite_absorbent_pareil() -> None:
    """Ce que l'opacite sert a dire, et la raison pour laquelle on la pose.

    Le coefficient etant proportionnel a `p`, l'exposant de Beer vaut
    `k p l` : il ne depend que de l'opacite. Deux melanges de meme `p l`
    absorbent donc identiquement, ce qui est exactement ce qui permet de
    lire les abaques en fonction d'une seule variable.
    """
    premier = absorptivite_du_trajet(
        coefficient=coefficient_lineaire(constante=3.0, pression_partielle=0.25),
        longueur=4.0,
    )
    second = absorptivite_du_trajet(
        coefficient=coefficient_lineaire(constante=3.0, pression_partielle=1.0),
        longueur=1.0,
    )
    assert premier == pytest.approx(second)


def test_la_longueur_moyenne_d_une_sphere_vaut_les_deux_tiers_du_diametre() -> None:
    """`4 V / S` sur une sphere donne `4 R / 3` : identite verifiable.

    Le syllabus donne le rayon moyen equivalent en abaque (4.4) et traite
    le volume hemispherique a part (4.3.1). L'expression `4 V / S` est la
    forme classique, et elle n'est pas prise sur parole ici : on la
    verifie sur la geometrie dont le resultat est connu.
    """
    rayon = 0.6
    volume = 4 / 3 * math.pi * rayon**3
    surface = 4 * math.pi * rayon**2
    assert longueur_moyenne_de_rayonnement(volume, surface) == pytest.approx(
        4 * rayon / 3,
    )


def test_la_longueur_moyenne_d_un_cube_vaut_les_deux_tiers_de_l_arete() -> None:
    """Seconde geometrie, seconde verification : `4 a^3 / 6 a^2 = 2 a / 3`."""
    arete = 1.5
    assert longueur_moyenne_de_rayonnement(arete**3, 6 * arete**2) == pytest.approx(
        2 * arete / 3,
    )


def test_un_coefficient_d_absorption_negatif_est_refuse() -> None:
    """Un gaz qui amplifierait le rayonnement n'existe pas dans ce cours."""
    with pytest.raises(ValueError, match="coefficient d'absorption negatif"):
        transmittance(coefficient=-0.1, longueur=1.0)


def test_un_trajet_negatif_est_refuse() -> None:
    """Un trajet negatif rendrait une transmittance au-dessus de l'unite."""
    with pytest.raises(ValueError, match="longueur de trajet negative"):
        transmittance(coefficient=0.3, longueur=-1.0)


def test_une_pression_partielle_negative_est_refusee() -> None:
    """L'opacite serait negative, et l'abaque ne se lit pas la."""
    with pytest.raises(ValueError, match="pression partielle negative"):
        opacite(pression_partielle=-0.2, longueur=1.0)


def test_une_surface_nulle_est_refusee_pour_la_longueur_moyenne() -> None:
    """Un volume sans surface n'echange avec rien : la division n'a pas de sens."""
    with pytest.raises(ValueError, match="surface d'echange nulle ou negative"):
        longueur_moyenne_de_rayonnement(1.0, 0.0)
