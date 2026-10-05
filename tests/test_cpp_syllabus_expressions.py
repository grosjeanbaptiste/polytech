"""Syllabus « Introduction a la programmation en C++ », chapitre 2 : expressions.

Source : `ProgrammationAlgorithmique/SYLLABUS/
Syllabus_Introduction_a_la_programmation_2015.pdf`, pages 11 a 14 (2.9,
« Operateurs et expressions ») et 22 (l'operateur ternaire), annexe A.

Chaque valeur a ete confirmee en compilant l'expression (clang++ 21).
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.caracteres import caractere_de_code
from polytech.algorithmique.domain.expressions import evaluer
from polytech.algorithmique.domain.incrementation import postfixe, prefixe
from polytech.algorithmique.domain.priorites import PRIORITES, calcul_binaire
from polytech.algorithmique.domain.types_cpp import affecter


@pytest.mark.parametrize(
    ("expression", "attendu"),
    [
        ("5 + 3 * -7", -16),
        ("5 - 2 / -1 + 8", 15),
        ("(5 + 3) * (-7)", -56),
        ("(5 + 10) / (-1 + 8)", 2),
    ],
)
def test_les_quatre_expressions_de_la_page_11(expression: str, attendu: int) -> None:
    """Les unaires d'abord, puis `* / %`, puis `+ -` ; `15/7` en entiers vaut 2."""
    assert evaluer(expression) == attendu


def test_la_division_entiere_et_la_division_reelle_de_la_page_11() -> None:
    """`7%2` vaut 1, `7/2` vaut 3 ; `7.0/2`, `7/2.0` et `7.0/2.0` valent 3.5."""
    assert (evaluer("7 % 2"), evaluer("7 / 2")) == (1, 3)
    assert {evaluer(e) for e in ("7.0 / 2", "7 / 2.0", "7.0 / 2.0")} == {3.5}


def test_le_type_des_operandes_decide_de_la_division() -> None:
    """Page 11 : `k = i / j` vaut 1, `h = f / g` vaut 1.5."""
    assert evaluer("i / j", {"i": 6, "j": 4}) == 1
    assert evaluer("f / g", {"f": 6.0, "g": 4.0}) == 1.5


def test_une_expression_a_variables_reelles() -> None:
    """Page 12 : `(3*x*x + 5*x*y + 7)/(x + 1)`."""
    assert evaluer("(3*x*x + 5*x*y + 7)/(x + 1)", {"x": 1.0, "y": 2.0}) == 10.0


def test_les_expressions_logiques_de_l_exemple_6() -> None:
    """Page 13 : `f > 5` est vrai, `(i + f) <= 1` est faux."""
    variables = {"i": 7, "f": 5.5, "val": -10}
    assert evaluer("f > 5", variables) is True
    assert evaluer("(i + f) <= 1", variables) is False
    assert evaluer("(val > 0) || (val == -10)", variables) is True
    assert evaluer("(val >= 0) && (val < 5)", variables) is False


def test_l_evaluation_paresseuse_de_la_page_13() -> None:
    """`expr2` n'est evaluee que si `expr1` ne suffit pas a conclure."""
    assert evaluer("0 && 1 / 0") is False
    assert evaluer("1 || 1 / 0") is True
    with pytest.raises(ZeroDivisionError, match="division par zero"):
        evaluer("1 && 1 / 0")
    with pytest.raises(ZeroDivisionError, match="division par zero"):
        evaluer("0 || 1 / 0")


def test_les_classes_de_priorite_de_la_page_14() -> None:
    """`!` avant `* / %`, avant `+ -`, avant `<`, avant `==`, avant `&&`, avant `||`."""
    assert evaluer("!0 + 1") == 2
    assert evaluer("1 + 2 < 4 == 1") is True
    assert evaluer("1 || 0 && 0") is True
    assert evaluer("2 < 3 == 3 < 2") is False
    assert evaluer("2 + 3 * 4 % 5") == 4


def test_la_table_des_priorites_va_du_moins_au_plus_prioritaire() -> None:
    """Les classes 8 a 3 de la page 14, dans l'ordre ou l'analyse les traverse."""
    assert PRIORITES[0] == ("||",)
    assert PRIORITES[1] == ("&&",)
    assert PRIORITES[-1] == ("*", "/", "%")
    assert len({op for classe in PRIORITES for op in classe}) == 13


def test_les_operateurs_d_une_meme_classe_vont_de_gauche_a_droite() -> None:
    """Page 14 : « l'evaluation se fait en passant de la gauche vers la droite »."""
    assert evaluer("8 / 4 / 2") == 1
    assert evaluer("8 - 4 - 2") == 2
    assert evaluer("7 % 4 * 2") == 6


def test_les_unaires_s_evaluent_de_droite_a_gauche() -> None:
    """`- - 3` vaut 3, `!!5` est vrai, `+ - 2` vaut -2."""
    assert evaluer("- - 3") == 3
    assert evaluer("!!5") is True
    assert evaluer("+ - 2") == -2


def test_l_operateur_ternaire_de_la_page_22() -> None:
    """`max = (a > b) ? a : b;` : a si a est plus grand que b, b sinon."""
    assert evaluer("(a > b) ? a : b", {"a": 3, "b": 7}) == 7
    assert evaluer("(a > b) ? a : b", {"a": 9, "b": 7}) == 9
    assert evaluer("0 ? 1 : 0 ? 2 : 3") == 3
    assert evaluer("1 ? 1 / 1 : 1 / 0") == 1


def test_l_exercice_3_1_un_test_toujours_vrai() -> None:
    """`choix == 'o' || 'O'` : le code de `O`, non nul, est vrai a lui seul."""
    for choix in (ord("o"), ord("x")):
        assert evaluer("choix == 111 || 79", {"choix": choix}) is True
    assert evaluer("choix == 111 || choix == 79", {"choix": ord("x")}) is False


def test_une_variable_non_declaree_est_refusee() -> None:
    """« Tout objet C++ doit etre declare avant d'etre utilise »."""
    with pytest.raises(NameError, match="variable non declaree"):
        evaluer("x + 1")


@pytest.mark.parametrize(
    "expression", ["5 +", "(5 + 3", "5 5", "5 $ 3", "", "1 ? 2", "* 3"]
)
def test_une_expression_mal_formee_est_refusee(expression: str) -> None:
    """Un operande manquant, une parenthese ouverte, un caractere inconnu."""
    with pytest.raises(SyntaxError, match="expression mal formee"):
        evaluer(expression)


def test_les_increments_de_l_exemple_7() -> None:
    """Page 14 : `i=4, j=3` ; `i++`, `--j`, `k = ++i` (6, 6), `k = j--` (2, 1)."""
    _, i = postfixe(4)
    _, j = prefixe(3, -1)
    assert (i, j) == (5, 2)
    k, i = prefixe(i)
    assert (k, i) == (6, 6)
    k, j = postfixe(j, -1)
    assert (k, j) == (2, 1)


def test_prefixe_rend_la_valeur_mise_a_jour_et_postfixe_l_ancienne() -> None:
    """Page 14 : `X = I++` passe d'abord la valeur ; `X = ++I` incremente d'abord."""
    assert postfixe(7) == (7, 8)
    assert prefixe(7) == (8, 8)
    assert postfixe(7, -1) == (7, 6)
    assert prefixe(7, -1) == (6, 6)


def test_les_conversions_forcees_de_l_exemple_5() -> None:
    """Page 12 : `(int) 65.49` vaut 65, `(char) 65.49` vaut A, `(char) 98` vaut b."""
    assert affecter(65.49, "int") == 65
    assert caractere_de_code(65.49) == "A"
    assert affecter(98, "float") == 98.0
    assert caractere_de_code(98) == "b"


def test_la_conversion_de_l_exemple_4() -> None:
    """Page 12 : `K = (float)I/J` avec I = 3 et J = 4 vaut 0.75, et non 0."""
    assert evaluer("I / J", {"I": 3.0, "J": 4}) == 0.75
    assert evaluer("I / J", {"I": 3, "J": 4}) == 0


def test_un_code_negatif_n_est_pas_un_caractere_affichable() -> None:
    """`(char) 200` deborde en -56 : hors de la table ASCII."""
    with pytest.raises(ValueError, match="hors de la table ASCII"):
        caractere_de_code(200)


def test_une_variable_peut_commencer_par_un_souligne() -> None:
    """Page 7 : « le caractere _ est considere comme une lettre »."""
    assert (
        evaluer("_position + fin_de_fichier", {"_position": 2, "fin_de_fichier": 3})
        == 5
    )


def test_un_calcul_binaire_ne_declenche_pas_l_operande_inutile() -> None:
    """`faux && ...` : l'operande de droite n'est jamais appele."""
    appels: list[str] = []

    def droite() -> bool:
        appels.append("droite")
        return True

    assert calcul_binaire("&&", lambda: False, droite)() is False
    assert calcul_binaire("||", lambda: True, droite)() is True
    assert appels == []
    assert calcul_binaire("&&", lambda: True, droite)() is True
    assert appels == ["droite"]


def test_le_modulo_a_le_signe_du_dividende_et_refuse_les_reels() -> None:
    """Page 11 : « % modulo (reste d'une div. entiere) » -- `-7 % 2` vaut -1."""
    assert evaluer("-7 % 2") == -1
    assert evaluer("7 % -2") == 1
    with pytest.raises(TypeError, match="modulo de deux entiers seulement"):
        evaluer("7.5 % 2")


def test_les_comparaisons_larges_et_la_difference() -> None:
    """Page 12 : `<=`, `>=` et `!=`, a ne pas confondre avec `<`, `>` et `==`."""
    assert evaluer("2 <= 2") is True
    assert evaluer("2 >= 2") is True
    assert evaluer("2 != 3") is True
    assert evaluer("2 != 2") is False


def test_un_caractere_inconnu_est_nomme() -> None:
    """Le refus dit quel caractere n'appartient pas au langage."""
    with pytest.raises(SyntaxError, match="caractere inattendu"):
        evaluer("5 $ 3")


@pytest.mark.parametrize("expression", [")", "5 + )", "5 + ?"])
def test_un_operateur_n_est_pas_un_operande(expression: str) -> None:
    """La ou une valeur est attendue, une parenthese fermante ne convient pas."""
    with pytest.raises(SyntaxError, match="un operande est attendu"):
        evaluer(expression)


@pytest.mark.parametrize(
    ("caractere", "decimal", "hexa", "octal"),
    [("A", 65, "0x41", "101"), ("a", 97, "0x61", "141"), ("0", 48, "0x30", "60")],
)
def test_les_lignes_de_la_table_ascii_de_l_annexe(
    caractere: str, decimal: int, hexa: str, octal: str
) -> None:
    """Annexe A, page 108 : le caractere, son code decimal, hexadecimal, octal."""
    assert caractere_de_code(decimal) == caractere
    assert (f"{decimal:#04x}", f"{decimal:o}") == (hexa, octal)
