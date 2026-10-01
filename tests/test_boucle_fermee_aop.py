"""Le chapitre 2 partie I : ce que la retroaction fait aux trois grandeurs.

Le fascicule enseigne une seule idee, declinee sur trois grandeurs : le
**rapport de boucle** `1 + A beta` divise le gain, multiplie l'impedance
d'entree, divise l'impedance de sortie, et divise la distorsion non
lineaire. Tout le reste du chapitre en decoule -- l'inverseur, le
non-inverseur, l'effet d'une resistance de source ou de charge.

Les ancrages sont exacts et internes. Le gain en boucle fermee doit tendre
vers `1/beta` quand le gain en boucle ouverte tend vers l'infini : c'est
ce qui rend un montage insensible a l'AOP. Et le produit gain-bande doit
etre **conserve** par la retroaction -- la planche y consacre trois
planches dont une preuve, et c'est une egalite, donc testable.
"""

from __future__ import annotations

import math

import pytest

from polytech.electronique.domain.boucle_fermee import (
    bande_apres_retroaction,
    coefficient_de_retour,
    distorsion_apres_retroaction,
    gain_en_boucle_fermee,
    impedance_d_entree_en_boucle_fermee,
    impedance_de_sortie_en_boucle_fermee,
    produit_gain_bande,
    rapport_de_boucle,
    retroaction_efficace,
)

GAIN_OUVERT = 1e5
RETOUR = 0.01


def test_le_rapport_de_boucle_est_un_plus_a_beta() -> None:
    """La seule quantite du chapitre : tout le reste s'en deduit."""
    assert rapport_de_boucle(GAIN_OUVERT, RETOUR) == pytest.approx(1 + 1e3)


def test_un_rapport_de_boucle_sous_l_unite_signale_une_reaction_positive() -> None:
    """`1 + A beta < 1` veut dire que la boucle **amplifie** l'ecart.

    Ce n'est plus une contre-reaction mais une reaction positive, et tous
    les resultats du chapitre s'inversent. Le dire vaut mieux que de
    rendre une impedance d'entree qui aurait diminue.
    """
    with pytest.raises(ValueError, match="reaction positive"):
        rapport_de_boucle(GAIN_OUVERT, -0.01)


def test_le_gain_en_boucle_fermee_tend_vers_l_inverse_du_retour() -> None:
    """C'est ce qui rend un montage insensible a l'AOP.

    Quand `A beta` est grand devant un, `A / (1 + A beta)` tend vers
    `1 / beta` -- le gain ne depend plus que du reseau de retour, donc de
    deux resistances et non des caprices du composant.
    """
    ferme = gain_en_boucle_fermee(GAIN_OUVERT, RETOUR)
    assert ferme == pytest.approx(1 / RETOUR, rel=1e-3)


def test_un_gain_de_boucle_ouverte_faible_ne_suffit_pas() -> None:
    """Avec `A beta` proche de un, le gain depend encore de l'AOP."""
    ferme = gain_en_boucle_fermee(10.0, RETOUR)
    assert ferme != pytest.approx(1 / RETOUR, rel=0.1)


def test_la_retroaction_multiplie_l_impedance_d_entree() -> None:
    """Serie en entree : l'impedance est multipliee par le rapport de boucle."""
    ouverte = 1e6
    fermee = impedance_d_entree_en_boucle_fermee(ouverte, GAIN_OUVERT, RETOUR)
    assert fermee == pytest.approx(ouverte * rapport_de_boucle(GAIN_OUVERT, RETOUR))


def test_la_retroaction_divise_l_impedance_de_sortie() -> None:
    """Parallele en sortie : elle est divisee par le meme rapport."""
    ouverte = 100.0
    fermee = impedance_de_sortie_en_boucle_fermee(ouverte, GAIN_OUVERT, RETOUR)
    assert fermee == pytest.approx(ouverte / rapport_de_boucle(GAIN_OUVERT, RETOUR))


def test_la_retroaction_divise_la_distorsion_non_lineaire() -> None:
    """« Other benefits » de la planche : le meme facteur, encore."""
    sans = 0.05
    avec = distorsion_apres_retroaction(sans, GAIN_OUVERT, RETOUR)
    assert avec == pytest.approx(sans / rapport_de_boucle(GAIN_OUVERT, RETOUR))


def test_le_coefficient_de_retour_se_lit_sur_le_diviseur() -> None:
    """`beta = R_1 / (R_1 + R_f)` pour le non-inverseur."""
    assert coefficient_de_retour(1e3, 99e3) == pytest.approx(0.01)


def test_un_reseau_de_retour_sans_resistance_est_refuse() -> None:
    with pytest.raises(ValueError, match="reseau de retour"):
        coefficient_de_retour(0.0, 0.0)


def test_le_produit_gain_bande_est_conserve_par_la_retroaction() -> None:
    """Trois planches et une preuve y sont consacrees : c'est une egalite.

    Le gain chute du rapport de boucle, la bande monte du meme rapport, et
    leur produit ne bouge pas. C'est ce qui permet de lire la bande d'un
    montage sur la fiche technique de l'AOP.
    """
    bande_ouverte = 10.0
    produit_ouvert = produit_gain_bande(GAIN_OUVERT, bande_ouverte)
    ferme = gain_en_boucle_fermee(GAIN_OUVERT, RETOUR)
    bande_fermee = bande_apres_retroaction(GAIN_OUVERT, bande_ouverte, RETOUR)
    assert produit_gain_bande(ferme, bande_fermee) == pytest.approx(produit_ouvert)


def test_la_bande_s_elargit_du_rapport_de_boucle() -> None:
    bande = bande_apres_retroaction(GAIN_OUVERT, 10.0, RETOUR)
    assert bande == pytest.approx(10.0 * rapport_de_boucle(GAIN_OUVERT, RETOUR))


def test_un_produit_gain_bande_sur_une_bande_nulle_est_refuse() -> None:
    with pytest.raises(ValueError, match="bande nulle"):
        produit_gain_bande(GAIN_OUVERT, 0.0)


def test_la_retroaction_est_efficace_quand_le_gain_de_boucle_est_grand() -> None:
    """« General conditions for a successful retroaction » de la planche.

    La condition est `A beta >> 1`. Le seuil retenu est cent, et il est
    dit : au-dela, l'ecart entre `A/(1+A beta)` et `1/beta` tombe sous le
    pour cent.
    """
    assert retroaction_efficace(GAIN_OUVERT, RETOUR)
    assert not retroaction_efficace(10.0, RETOUR)


def test_l_ecart_au_gain_idealise_tombe_sous_le_pour_cent_au_seuil() -> None:
    """Le seuil de cent n'est pas arbitraire : il est verifie ici.

    C'est ce qui le distingue d'un nombre choisi pour que le test passe.
    """
    retour = 0.01
    gain_au_seuil = 100 / retour
    ferme = gain_en_boucle_fermee(gain_au_seuil, retour)
    ecart = abs(ferme - 1 / retour) / (1 / retour)
    assert ecart < 0.01
    assert retroaction_efficace(gain_au_seuil, retour)


def test_le_gain_en_decibels_du_montage_non_inverseur() -> None:
    """Un gain de cent fait quarante decibels : rapport de **tensions**."""
    gain = gain_en_boucle_fermee(GAIN_OUVERT, RETOUR)
    assert 20 * math.log10(gain) == pytest.approx(40.0, abs=0.01)
