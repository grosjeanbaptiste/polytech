"""La section 12.2 du syllabus : l'aimant optimal.

« Pour realiser une induction donnee `B_e` dans un entrefer donne de
volume `V_e`, on aura donc a utiliser le plus petit volume `V` d'un
materiau donne si le point de fonctionnement est tel que le produit
`-H B` passe par un maximum (`H` est negatif). » Et le syllabus donne la
lecture graphique : « le sommet de celui des rectangles inscrits dans la
courbe de desaimantation qui a l'aire maximale ».

Le resultat ferme n'est pas pris sur parole. Sur la courbe de
desaimantation **lineaire** -- le modele usuel entre l'induction
remanente et le champ coercitif -- le maximum se calcule, et le test le
compare a une **maximisation numerique** du produit `-H B` : deux chemins
sans primitive commune.

Le syllabus nomme enfin les trois specifications d'un materiau pour
aimant permanent : `(HB)max`, l'induction remanente `B_r` et le champ
coercitif `H_c`. Les trois sont ici, et les deux premieres se deduisent
des deux dernieres.
"""

from __future__ import annotations

import pytest

from polytech.electromagnetisme.domain.magnetostatique.aimants import (
    SPECIFICATIONS,
    induction_sur_la_droite_de_desaimantation,
    point_de_fonctionnement_optimal,
    produit_d_energie,
    produit_maximal,
    volume_minimal_de_l_aimant,
)

# Ferrite usuelle : induction remanente en teslas, champ coercitif en A/m.
REMANENTE = 0.4
COERCITIF = 2.4e5


def test_la_droite_de_desaimantation_joint_les_deux_specifications() -> None:
    """A champ nul l'induction vaut `B_r` ; a `-H_c` elle s'annule.

    C'est ce que « pour un cycle a saturation » designe, et c'est le
    modele le plus simple qui respecte les deux specifications que le
    syllabus nomme.
    """
    assert induction_sur_la_droite_de_desaimantation(0.0, REMANENTE, COERCITIF) == (
        pytest.approx(REMANENTE)
    )
    assert induction_sur_la_droite_de_desaimantation(
        -COERCITIF,
        REMANENTE,
        COERCITIF,
    ) == pytest.approx(0.0)


def test_le_point_optimal_est_a_la_moitie_des_deux_specifications() -> None:
    """`H = -H_c/2` et `B = B_r/2` : le sommet du plus grand rectangle.

    Le resultat est elegant et il n'est pas evident : l'aire maximale
    d'un rectangle inscrit sous une droite est atteinte au milieu, et
    c'est pourquoi un aimant bien dimensionne travaille a la moitie de
    son induction remanente -- non pres de celle-ci.
    """
    champ, induction = point_de_fonctionnement_optimal(REMANENTE, COERCITIF)
    assert champ == pytest.approx(-COERCITIF / 2)
    assert induction == pytest.approx(REMANENTE / 2)


def test_le_maximum_ferme_concorde_avec_une_maximisation_numerique() -> None:
    """Deux chemins : la forme fermee, et le balayage de la droite.

    Si le point optimal etait mal place, le balayage trouverait mieux --
    et c'est exactement ce que ce test interdit.
    """
    pas = COERCITIF / 2000
    meilleur = max(
        produit_d_energie(
            induction_sur_la_droite_de_desaimantation(
                -i * pas,
                REMANENTE,
                COERCITIF,
            ),
            -i * pas,
        )
        for i in range(2001)
    )
    assert produit_maximal(REMANENTE, COERCITIF) == pytest.approx(meilleur, rel=1e-4)


def test_le_produit_maximal_vaut_le_quart_du_produit_des_specifications() -> None:
    """`(HB)max = B_r H_c / 4` sur la droite de desaimantation.

    C'est la forme fermee du resultat precedent, et elle dit pourquoi les
    fiches techniques donnent `(HB)max` : il resume a lui seul la
    capacite du materiau, et il croit comme le **produit** des deux
    specifications.
    """
    assert produit_maximal(REMANENTE, COERCITIF) == pytest.approx(
        REMANENTE * COERCITIF / 4,
    )


def test_doubler_les_deux_specifications_quadruple_le_produit() -> None:
    """Et c'est ce qui separe une ferrite d'un aimant aux terres rares."""
    simple = produit_maximal(REMANENTE, COERCITIF)
    double = produit_maximal(2 * REMANENTE, 2 * COERCITIF)
    assert double == pytest.approx(4 * simple)


def test_le_volume_minimal_decroit_quand_le_materiau_est_meilleur() -> None:
    """« Le plus petit volume V d'un materiau donne », dit le syllabus.

    A entrefer fixe, le volume d'aimant necessaire est inversement
    proportionnel au produit d'energie du materiau : un aimant deux fois
    meilleur est deux fois plus petit, et c'est tout l'enjeu industriel
    du chapitre 12.3.
    """
    entrefer, volume_entrefer = 0.3, 1e-6
    ferrite = volume_minimal_de_l_aimant(
        entrefer,
        volume_entrefer,
        produit_maximal(REMANENTE, COERCITIF),
    )
    meilleur = volume_minimal_de_l_aimant(
        entrefer,
        volume_entrefer,
        2 * produit_maximal(REMANENTE, COERCITIF),
    )
    assert meilleur == pytest.approx(ferrite / 2)


def test_les_trois_specifications_que_le_syllabus_nomme() -> None:
    """`(HB)max`, l'induction remanente, le champ coercitif."""
    assert len(SPECIFICATIONS) == 3
    assert any("HB" in s for s in SPECIFICATIONS)
    assert any("remanente" in s for s in SPECIFICATIONS)
    assert any("coercitif" in s for s in SPECIFICATIONS)


def test_un_champ_positif_est_refuse_sur_la_droite_de_desaimantation() -> None:
    """« H est negatif », precise le syllabus entre parentheses.

    La courbe de desaimantation est le deuxieme quadrant du cycle ; un
    champ positif y est hors du domaine, et le produit `-H B` y serait
    negatif sans que rien ne le signale.
    """
    with pytest.raises(ValueError, match="champ positif"):
        induction_sur_la_droite_de_desaimantation(1e4, REMANENTE, COERCITIF)


def test_un_champ_au_dela_du_coercitif_est_refuse() -> None:
    """L'induction y serait negative : l'aimant est desaimante bien avant."""
    with pytest.raises(ValueError, match="au-dela du champ coercitif"):
        induction_sur_la_droite_de_desaimantation(
            -2 * COERCITIF,
            REMANENTE,
            COERCITIF,
        )


def test_un_produit_d_energie_nul_refuse_le_volume() -> None:
    """Un materiau sans produit d'energie n'est pas un aimant."""
    with pytest.raises(ValueError, match="produit d'energie nul ou negatif"):
        volume_minimal_de_l_aimant(0.3, 1e-6, 0.0)
