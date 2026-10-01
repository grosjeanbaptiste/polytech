"""Les codes de lecture des composants, et les huit exemples de la planche.

La planche « Presentation des labos » a l'air d'etre de la logistique --
horaires, sachets, protoboard -- et pour moitie elle l'est. Mais elle
enseigne aussi quatre lectures : la resistance a bagues de couleur, le
condensateur a code chiffre, le composant DIP et la diode. Les deux
premieres sont du **calcul**, et la planche donne ses propres oracles :
un tableau complet des couleurs, et huit marquages de condensateur avec
leur valeur.

Ces huit exemples sont les cas de test de ce fichier. Ils couvrent les
quatre formes que le marquage prend -- trois chiffres, prefixe collant
(`µ1`, `n10`), virgule initiale (`.22J`), et lettre de tolerance -- et
c'est ce qui en fait une bonne verification : un analyseur qui ne
traiterait que les trois chiffres echouerait sur cinq des huit.
"""

from __future__ import annotations

import pytest

from polytech.electronique.domain.code_des_composants import (
    TOLERANCES_LITTERALES,
    capacite_du_marquage,
    intervalle_de_tolerance,
    numero_oppose_sur_un_dip,
    polarite_de_la_patte_longue,
    rangee_de_la_patte,
    resistance_des_bagues,
    tolerance_de_la_bague,
)

PICO = 1e-12
NANO = 1e-9
MICRO = 1e-6


def test_les_trois_bagues_donnent_la_resistance_du_tableau() -> None:
    """« 4,7 k » et « 2,4 k » sont les deux valeurs que la planche montre.

    Jaune-violet-rouge vaut `47 x 100 = 4,7 k`, et rouge-jaune-rouge
    `24 x 100 = 2,4 k`. Les deux sortent du meme tableau, et ce sont les
    deux resistances dessinees sur la planche.
    """
    assert resistance_des_bagues("jaune", "violet", "rouge") == pytest.approx(4700.0)
    assert resistance_des_bagues("rouge", "jaune", "rouge") == pytest.approx(2400.0)


def test_le_noir_en_multiplicateur_ne_multiplie_pas() -> None:
    """« Noir : 0, x 1 » -- le zero d'exposant, qui surprend d'abord.

    Marron-noir-noir vaut donc 10 ohms et non 100 : la troisieme bague
    porte l'exposant, et l'exposant zero laisse la valeur intacte.
    """
    assert resistance_des_bagues("marron", "noir", "noir") == pytest.approx(10.0)


def test_le_tableau_s_arrete_au_megohm() -> None:
    """Violet, gris et blanc n'ont pas de multiplicateur dans ce tableau.

    La planche porte un tiret en face des trois. Les accepter en
    troisieme bague serait inventer trois decades que le tableau du cours
    ne donne pas -- et un etudiant qui lirait une telle bague devrait
    sortir l'ohmmetre, ce que la planche recommande d'ailleurs « dans le
    doute ».
    """
    with pytest.raises(ValueError, match="pas de multiplicateur"):
        resistance_des_bagues("marron", "noir", "blanc")


def test_les_quatre_tolerances_du_tableau() -> None:
    """Argent 10 %, or 5 %, rouge 2 %, marron 1 % -- et rien d'autre."""
    assert tolerance_de_la_bague("argent") == pytest.approx(0.10)
    assert tolerance_de_la_bague("or") == pytest.approx(0.05)
    assert tolerance_de_la_bague("rouge") == pytest.approx(0.02)
    assert tolerance_de_la_bague("marron") == pytest.approx(0.01)


def test_une_bague_de_tolerance_inconnue_est_refusee() -> None:
    """Le vert n'est pas une bague de tolerance, bien qu'il soit un chiffre.

    C'est le piege du tableau : les memes couleurs servent dans les deux
    colonnes, mais pas toutes. Rouge et marron sont dans les deux ; vert
    n'est que dans les chiffres.
    """
    with pytest.raises(ValueError, match="n'est pas une bague de tolerance"):
        tolerance_de_la_bague("vert")


@pytest.mark.parametrize(
    ("marquage", "attendue"),
    [
        ("104", 100 * NANO),
        ("123", 12 * NANO),
        ("µ1", 100 * NANO),
        ("22M", 22 * PICO),
        ("n10", 100 * PICO),
        (".22J", 220 * NANO),
        ("2µ3", 2.3 * MICRO),
        ("103F50", 10 * NANO),
    ],
)
def test_les_huit_marquages_de_la_planche(marquage: str, attendue: float) -> None:
    """Les huit exemples que la planche resout elle-meme, un par un.

    Ils sont l'oracle : la planche donne le marquage **et** sa valeur, si
    bien qu'il n'y a rien a supposer. Les quatre formes y sont -- trois
    chiffres, prefixe collant, virgule initiale, lettre de tolerance.
    """
    assert capacite_du_marquage(marquage) == pytest.approx(attendue, rel=1e-9)


def test_la_tolerance_litterale_se_lit_sur_le_marquage() -> None:
    """`M` vaut 20 %, `J` 5 %, `F` 1 % -- les trois lettres de la planche."""
    assert TOLERANCES_LITTERALES["M"] == pytest.approx(0.20)
    assert TOLERANCES_LITTERALES["J"] == pytest.approx(0.05)
    assert TOLERANCES_LITTERALES["F"] == pytest.approx(0.01)


def test_un_marquage_vide_est_refuse() -> None:
    """Un condensateur sans marquage se mesure, il ne se devine pas."""
    with pytest.raises(ValueError, match="marquage vide"):
        capacite_du_marquage("")


def test_les_pattes_d_un_dip_se_font_face_en_sommant_a_un_de_plus() -> None:
    """« Parcours des pattes dans le sens trigonometrique », dit la planche.

    Sur le boitier a huit pattes qu'elle dessine, 1 fait face a 8, 2 a 7,
    3 a 6, 4 a 5 : la somme vaut toujours neuf. C'est la regle qui evite
    de compter les pattes a l'envers, et l'erreur de cablage qui en
    resulte est celle qui grille un composant.
    """
    for patte, opposee in ((1, 8), (2, 7), (3, 6), (4, 5)):
        assert numero_oppose_sur_un_dip(patte, pattes=8) == opposee
        assert numero_oppose_sur_un_dip(opposee, pattes=8) == patte


def test_un_dip_a_nombre_impair_de_pattes_est_refuse() -> None:
    """Un boitier DIP a deux rangees egales : son compte est pair."""
    with pytest.raises(ValueError, match="nombre de pattes impair"):
        numero_oppose_sur_un_dip(1, pattes=7)


def test_une_patte_hors_du_boitier_est_refusee() -> None:
    """Demander la patte 9 d'un boitier a huit pattes est une faute de lecture."""
    with pytest.raises(ValueError, match="patte hors du boitier"):
        numero_oppose_sur_un_dip(9, pattes=8)


def test_l_intervalle_de_tolerance_encadre_la_valeur_nominale() -> None:
    """A quoi sert la quatrieme bague : dire ce que la resistance peut valoir.

    Une resistance de 4,7 k a 5 % vaut entre 4465 et 4935 ohms. C'est
    l'intervalle qui compte dans un dimensionnement, et non la valeur
    nominale -- un diviseur calcule au nominal peut sortir de ses
    specifications sur les bornes.
    """
    bas, haut = intervalle_de_tolerance(4700.0, tolerance_de_la_bague("or"))
    assert bas == pytest.approx(4465.0)
    assert haut == pytest.approx(4935.0)
    assert bas < 4700.0 < haut


def test_une_tolerance_plus_serree_resserre_l_intervalle() -> None:
    """Marron a 1 % contre argent a 10 % : un facteur dix sur la largeur."""
    _, serre = intervalle_de_tolerance(1000.0, tolerance_de_la_bague("marron"))
    _, large = intervalle_de_tolerance(1000.0, tolerance_de_la_bague("argent"))
    assert (large - 1000.0) / (serre - 1000.0) == pytest.approx(10.0)


def test_la_patte_longue_est_le_plus() -> None:
    """« Longue patte = +, cote avec un "-" = - », dit la planche.

    Elle le dit pour le condensateur polarise et pour la LED, et c'est
    l'erreur qui grille le composant au premier branchement -- un
    electrolytique a l'envers ne pardonne pas.
    """
    assert polarite_de_la_patte_longue() == "+"


def test_le_repere_designe_la_premiere_patte_et_la_rangee_superieure() -> None:
    """« Point ou encoche : cote superieur », et le parcours est trigonometrique.

    Sur le boitier a huit pattes, les pattes 1 a 4 sont donc en bas apres
    rotation... non : le parcours trigonometrique part du repere, descend
    la premiere rangee puis remonte la seconde. Les pattes 1 a 4 sont
    d'un cote, 5 a 8 de l'autre, et c'est ce que ce test fixe.
    """
    assert rangee_de_la_patte(1, pattes=8) == "premiere"
    assert rangee_de_la_patte(4, pattes=8) == "premiere"
    assert rangee_de_la_patte(5, pattes=8) == "seconde"
    assert rangee_de_la_patte(8, pattes=8) == "seconde"


def test_une_tolerance_negative_est_refusee() -> None:
    """Une tolerance negative inverserait les bornes de l'intervalle."""
    with pytest.raises(ValueError, match="tolerance negative"):
        intervalle_de_tolerance(1000.0, -0.05)


def test_une_couleur_absente_du_tableau_est_refusee() -> None:
    """Le tableau de la planche a dix couleurs, et pas une de plus.

    « Turquoise » n'y est pas, et le deviner serait inventer un chiffre.
    La planche dit quoi faire a la place quand on hesite : « dans le
    doute, utiliser un ohmmetre ».
    """
    with pytest.raises(ValueError, match="couleur inconnue au tableau"):
        resistance_des_bagues("turquoise", "violet", "rouge")
