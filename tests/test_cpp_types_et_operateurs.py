"""Programmation et algorithmique, partie 1 : types, variables, operateurs.

Source : `ProgrammationAlgorithmique/SLIDES/Slides_2023_2024_Partie_1.pdf`,
planches 11 a 26. Le tableau des types de la planche 11, les divisions de
la planche 24, les contractions et increments de la planche 26, la
conversion de casse de la planche 36, le debordement de la planche 102.

Chaque resultat affiche par une planche a ete confirme en compilant son
programme (clang++ 21, C++17) avant d'etre recopie ici.
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.types_cpp import (
    TYPES_ENTIERS,
    affecter,
    caractere_decale,
    diviser,
    est_vrai,
    modulo,
    tronquer,
)

ENTIER = TYPES_ENTIERS["int"]
CARACTERE = TYPES_ENTIERS["char"]


@pytest.mark.parametrize(
    ("nom", "minimum", "maximum"),
    [
        ("char", -128, 127),
        ("unsigned char", 0, 255),
        ("short int", -32_768, 32_767),
        ("unsigned short int", 0, 65_535),
        ("int", -2_147_483_648, 2_147_483_647),
        ("unsigned int", 0, 4_294_967_295),
        ("long int", -2_147_483_648, 2_147_483_647),
        ("unsigned long int", 0, 4_294_967_295),
    ],
)
def test_les_plages_de_valeurs_de_la_planche_11(
    nom: str, minimum: int, maximum: int
) -> None:
    """Chaque type entier, et la plage que la planche lui donne (int sur 32 bits)."""
    plage = TYPES_ENTIERS[nom]
    assert (plage.minimum, plage.maximum) == (minimum, maximum)


def test_la_plage_decoule_du_nombre_de_bits() -> None:
    """Signe : de `-2^(n-1)` a `2^(n-1) - 1` ; non signe : de 0 a `2^n - 1`."""
    for plage in TYPES_ENTIERS.values():
        if plage.signe:
            attendue = (-(2 ** (plage.bits - 1)), 2 ** (plage.bits - 1) - 1)
        else:
            attendue = (0, 2**plage.bits - 1)
        assert (plage.minimum, plage.maximum) == attendue


def test_la_division_de_deux_entiers_est_entiere() -> None:
    """Planche 24 : `7/2` vaut 3 ; `7.0/2`, `7/2.0` et `7.0/2.0` valent 3.5."""
    assert diviser(7, 2) == 3
    assert isinstance(diviser(7, 2), int)
    assert diviser(7.0, 2) == diviser(7, 2.0) == diviser(7.0, 2.0) == 3.5


def test_la_division_entiere_tronque_vers_zero() -> None:
    """En C++ `-7/2` vaut -3, et non -4 comme la division entiere de Python."""
    assert diviser(-7, 2) == -3
    assert diviser(7, -2) == -3
    assert diviser(-7, -2) == 3
    grand = 9_007_199_254_740_993
    assert diviser(grand, 1) == grand
    assert -7 // 2 == -4


def test_le_modulo_ne_s_applique_qu_aux_entiers() -> None:
    """Planche 24 : « % ne peut etre utilise qu'avec des entiers »."""
    assert modulo(7, 2) == 1
    assert modulo(-7, 2) == -1
    assert modulo(7, -2) == 1
    with pytest.raises(TypeError, match="modulo de deux entiers seulement"):
        modulo(7.0, 2)


def test_une_division_par_zero_n_a_pas_de_resultat() -> None:
    with pytest.raises(ZeroDivisionError, match="division par zero"):
        diviser(7, 0)
    with pytest.raises(ZeroDivisionError, match="division par zero"):
        modulo(7, 0)


def test_les_deux_programmes_de_la_planche_24() -> None:
    """`d = 7`, `t = 4` : `d/t` affiche 1 ; `V*t` vaut 7.2, et `d = V*t` vaut 7."""
    assert diviser(7, 4) == 1
    assert affecter(diviser(7, 4), "float") == 1.0
    produit = 1.8 * 4
    assert produit == pytest.approx(7.2)
    assert affecter(produit, "int") == 7


def test_affecter_un_reel_a_un_entier_tronque_et_n_arrondit_pas() -> None:
    """`int n = 7.9;` range 7, et `-7.9` range -7."""
    assert affecter(7.9, "int") == 7
    assert affecter(-7.9, "int") == -7


def test_la_conversion_en_entier_tronque() -> None:
    """Planche 105 : `(int)8.85` vaut 8 ; la troncature va vers zero."""
    assert tronquer(8.85) == 8
    assert tronquer(-8.85) == -8
    assert tronquer(2.0) == 2


def test_un_entier_trop_grand_deborde() -> None:
    """Planche 102 : `3^21` calcule dans un `int` affiche 1870418611."""
    puissance = 1
    for _ in range(21):
        puissance = ENTIER.ranger(puissance * 3)
    assert puissance == 1_870_418_611
    assert 3**21 == 10_460_353_203
    assert ENTIER.ranger(5**3) == 125
    assert ENTIER.ranger(7**9) == 40_353_607


def test_le_debordement_replie_la_valeur_dans_la_plage() -> None:
    """Complement a deux : `2147483647 + 1` devient `-2147483648`."""
    assert ENTIER.ranger(2_147_483_647 + 1) == -2_147_483_648
    assert ENTIER.ranger(-2_147_483_648 - 1) == 2_147_483_647
    assert TYPES_ENTIERS["unsigned char"].ranger(256) == 0
    assert TYPES_ENTIERS["unsigned char"].ranger(-1) == 255
    assert ENTIER.contient(2_147_483_647)
    assert not ENTIER.contient(2_147_483_648)


def test_la_valeur_de_remplissage_d_une_variable_non_initialisee() -> None:
    """Planche 117 : `-858993460`, soit `0xCCCCCCCC` lu comme un `int`."""
    assert ENTIER.ranger(0xCCCCCCCC) == -858_993_460


def test_les_increments_de_la_planche_26() -> None:
    """`i++`, `--j`, `++i` ; et `lettre++` qui fait passer de B a C."""
    assert caractere_decale("B", 1) == "C"
    assert caractere_decale("C", -1) == "B"
    i, j = 5, 4
    i += 1
    j -= 1
    i += 1
    assert (i, j) == (7, 3)


def test_la_conversion_de_casse_de_la_planche_36() -> None:
    """`b = a + ('a' - 'A')` : l'ecart entre minuscule et majuscule est 32."""
    ecart = ord("a") - ord("A")
    assert ecart == 32
    assert caractere_decale("G", ecart) == "g"
    assert caractere_decale("g", -ecart) == "G"
    paires = [(c, caractere_decale(c, 32)) for c in "ABCDE"]
    assert paires == [("A", "a"), ("B", "b"), ("C", "c"), ("D", "d"), ("E", "e")]


def test_un_caractere_hors_du_type_char_est_refuse() -> None:
    with pytest.raises(ValueError, match="hors de la plage"):
        caractere_decale("z", 200)


def test_toute_valeur_non_nulle_est_vraie() -> None:
    """Planche 25 : « toute valeur non nulle est consideree comme vraie »."""
    assert est_vrai(1)
    assert est_vrai(-2)
    assert est_vrai(0.5)
    assert not est_vrai(0)
    assert not est_vrai(0.0)


def test_une_affectation_dans_un_test_vaut_la_valeur_affectee() -> None:
    """Planche 34 : `if (i = -2)` est toujours vrai, quel que soit `i`."""
    valeur_affectee = -2
    assert est_vrai(valeur_affectee)
    comparaison = -1 == -2
    assert not est_vrai(int(comparaison))


def test_un_type_inconnu_est_refuse() -> None:
    with pytest.raises(ValueError, match="type inconnu"):
        affecter(3, "quaternion")
