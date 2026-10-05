"""Programmation et algorithmique, partie 1 : `cout`, le formatage, les identificateurs.

Source : `ProgrammationAlgorithmique/SLIDES/Slides_2023_2024_Partie_1.pdf`,
planches 10, 14 et 15 a 19. Les identificateurs valides et invalides de la
planche 10, les variables auto-typees de la planche 14, les caracteres
d'echappement de la planche 17, `setprecision`, `setw` et `setfill` aux
planches 18 et 19.

Chaque affichage a ete confirme en compilant le programme de la planche
(clang++ 21, C++17). La compilation etablit aussi un ecart : planche 14,
`auto K, L;` n'est pas du C++ -- une variable `auto` exige une
initialisation.
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.flux_de_sortie import (
    ECHAPPEMENTS,
    Ecran,
    ecrire_un_reel,
)
from polytech.algorithmique.domain.identificateurs import (
    MOTS_RESERVES,
    est_un_identificateur,
    motif_de_refus,
    type_deduit,
)

PI = 3.14159265359


def test_l_affichage_d_un_reel_par_defaut() -> None:
    """Six chiffres significatifs, et pas de zeros inutiles : `7.2`, `3`, `0.333333`."""
    assert ecrire_un_reel(PI) == "3.14159"
    assert ecrire_un_reel(7.2) == "7.2"
    assert ecrire_un_reel(3.0) == "3"
    assert ecrire_un_reel(1 / 3) == "0.333333"
    assert ecrire_un_reel(116.11) == "116.11"


def test_un_reel_trop_grand_ou_trop_petit_passe_en_notation_scientifique() -> None:
    """`1234567.0` s'affiche `1.23457e+06`, et `0.000012345`, `1.2345e-05`."""
    assert ecrire_un_reel(1234567.0) == "1.23457e+06"
    assert ecrire_un_reel(0.000012345) == "1.2345e-05"
    assert ecrire_un_reel(123456.7) == "123457"


def test_setprecision_fixe_le_nombre_de_chiffres_significatifs() -> None:
    """Planche 18 : `3.14159265` pour 9, `3.1` pour 2."""
    assert ecrire_un_reel(PI, precision=9) == "3.14159265"
    assert ecrire_un_reel(PI, precision=2) == "3.1"
    assert ecrire_un_reel(1234.5, precision=3) == "1.23e+03"


def test_une_precision_nulle_ou_negative_est_refusee() -> None:
    with pytest.raises(ValueError, match="precision nulle ou negative"):
        ecrire_un_reel(PI, precision=-1)
    with pytest.raises(ValueError, match="precision nulle ou negative"):
        ecrire_un_reel(PI, precision=0)


def test_le_programme_de_la_planche_18() -> None:
    """Cinq lignes, dont `.....3.14159265` : `setfill` remplit la largeur de `setw`."""
    ecran = Ecran()
    ecran.ecrire(PI).endl()
    ecran.setprecision(9).ecrire(PI).endl()
    ecran.setprecision(2).ecrire(PI).endl()
    ecran.setw(10).setprecision(2).ecrire(PI).endl()
    ecran.setw(15).setfill(".").setprecision(9).ecrire(PI).endl()
    assert ecran.lignes() == [
        "3.14159",
        "3.14159265",
        "3.1",
        "       3.1",
        ".....3.14159265",
    ]


def test_setw_ne_vaut_que_pour_la_sortie_suivante() -> None:
    """Planche 19 : `setw(7) << 1 << setw(10) << 2 << setw(5) << 3`."""
    ecran = Ecran()
    ecran.setw(7).ecrire(1).setw(10).ecrire(2).setw(5).ecrire(3)
    assert ecran.texte == "      1         2    3"
    suite = Ecran().setw(4).ecrire(12).ecrire(34)
    assert suite.texte == "  1234"


def test_setprecision_et_setfill_restent_en_vigueur() -> None:
    """A l'inverse de `setw` : la precision et le remplissage persistent."""
    ecran = Ecran().setprecision(2).setfill("*")
    ecran.ecrire(PI).ecrire(" ").setw(5).ecrire(PI).ecrire(" ").setw(5).ecrire(PI)
    assert ecran.texte == "3.1 **3.1 **3.1"


def test_une_largeur_trop_petite_ne_tronque_rien() -> None:
    """`setw(2) << "abcd"` affiche `abcd` : la largeur est un minimum."""
    assert Ecran().setw(2).ecrire("abcd").texte == "abcd"
    assert Ecran().setw(6).ecrire("ab").texte == "    ab"


def test_un_booleen_s_affiche_comme_un_entier() -> None:
    """Planche 11 : « une conversion implicite valant 0 ou 1 »."""
    vrai, faux = True, False
    assert Ecran().ecrire(vrai).ecrire(faux).texte == "10"


def test_le_texte_de_la_planche_17() -> None:
    """Une tabulation et un saut de ligne dans la chaine, puis `endl` : trois lignes."""
    ecran = Ecran()
    ecran.ecrire("Qu'il est agreable d'utiliser ")
    ecran.ecrire("cout en\t C++,\nlorsqu'on l'utilise ").endl().ecrire("\t\tproprement")
    assert ecran.lignes() == [
        "Qu'il est agreable d'utiliser cout en\t C++,",
        "lorsqu'on l'utilise ",
        "\t\tproprement",
    ]


def test_les_caracteres_d_echappement_de_la_planche_17() -> None:
    """Six caracteres, et le code ASCII que la planche donne en hexadecimal."""
    assert ECHAPPEMENTS == {
        "\\n": 0x0A,
        "\\t": 0x09,
        "\\v": 0x0B,
        "\\b": 0x08,
        "\\r": 0x0D,
        "\\f": 0x0C,
    }
    assert ECHAPPEMENTS["\\n"] == ord("\n")
    assert ECHAPPEMENTS["\\t"] == ord("\t")
    assert ECHAPPEMENTS["\\r"] == ord("\r")


def test_le_programme_de_la_planche_16() -> None:
    """`X = 5` : « Actuellement X vaut:5 et X*X vaut: 25 », puis « - X vaut : -5 »."""
    x = 5
    ecran = Ecran()
    ecran.ecrire(" Actuellement X vaut:").ecrire(x).ecrire(" et X*X vaut: ")
    ecran.ecrire(x * x).endl().ecrire(" - X vaut : ").ecrire(-x)
    assert ecran.lignes() == [
        " Actuellement X vaut:5 et X*X vaut: 25",
        " - X vaut : -5",
    ]


def test_le_premier_programme_de_la_planche_22() -> None:
    """`Y = X^2`, `Z = X^3` et la racine : pour `X = 5`, 25, 125 et 2.23607."""
    x = 5
    y = x * x
    z = y * x
    ecran = Ecran().ecrire("La racine de").ecrire(x).ecrire(" est ").ecrire(x**0.5)
    ecran.endl().ecrire(" Y = ").ecrire(y).ecrire(" et Z = ").ecrire(z)
    assert ecran.lignes() == ["La racine de5 est 2.23607", " Y = 25 et Z = 125"]


@pytest.mark.parametrize(
    "nom", ["xx", "y1", "somme_5", "_position", "Noms", "Prog_1_V0", "AB", "Ab", "ab"]
)
def test_les_identificateurs_valides_de_la_planche_10(nom: str) -> None:
    assert est_un_identificateur(nom)
    assert motif_de_refus(nom) is None


@pytest.mark.parametrize(
    ("nom", "motif"),
    [
        ("3eme", "commence par un chiffre"),
        ("x#y", "caractere non autorise (#)"),
        ("no-commande", "caractere non autorise (-)"),
        ("taux change", "caractere non autorise (espace)"),
    ],
)
def test_les_identificateurs_invalides_de_la_planche_10(nom: str, motif: str) -> None:
    """Chacun avec la raison que la planche donne."""
    assert not est_un_identificateur(nom)
    assert motif_de_refus(nom) == motif


def test_le_c_plus_plus_distingue_les_majuscules() -> None:
    """Planche 10 : `AB`, `Ab` et `ab` sont trois identificateurs differents."""
    assert len({"AB", "Ab", "ab"}) == 3
    assert est_un_identificateur("While")
    assert not est_un_identificateur("while")


def test_un_mot_reserve_n_est_pas_un_identificateur() -> None:
    """Planche 5 : les mots reserves, tous en minuscules."""
    for mot in ("int", "double", "return", "class", "switch", "nullptr"):
        assert mot in MOTS_RESERVES
        assert motif_de_refus(mot) == "mot reserve du langage"
    assert all(mot == mot.lower() for mot in MOTS_RESERVES)
    assert motif_de_refus("") == "identificateur vide"


def test_les_variables_auto_typees_de_la_planche_14() -> None:
    """`auto j = 0;` est un `int` ; un reel est un `double`."""
    assert type_deduit(0) == "int"
    assert type_deduit(1.0) == "double"
    assert type_deduit(2.2) == "double"
    assert type_deduit("A", guillemets="'") == "char"
    assert type_deduit("texte") == "const char*"
    booleen = True
    assert type_deduit(booleen) == "bool"


def test_auto_exige_une_initialisation() -> None:
    """Planche 14 : `auto K, L;` ne compile pas, contrairement a ce qu'elle montre."""
    with pytest.raises(ValueError, match="auto exige une initialisation"):
        type_deduit(None)
