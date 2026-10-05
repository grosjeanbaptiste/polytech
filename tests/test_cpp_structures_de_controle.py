"""Programmation et algorithmique, partie 1 : les structures de controle.

Source : `ProgrammationAlgorithmique/SLIDES/Slides_2023_2024_Partie_1.pdf`,
planches 27 a 53. Les decisions (`if`, `else`, `if` emboites), les
iterations (`for`, `while`, `do while`), les instructions d'echappement
(`break`, `continue`) et le choix multiple (`switch`).

Chaque affichage a ete confirme en compilant le programme de la planche
(clang++ 21, C++17). La compilation etablit aussi un ecart : planche 53,
il manque un point-virgule devant le second `break`, et le programme ne
compile pas.
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.controle import (
    BoucleInfinie,
    Cas,
    boucle_do_while,
    boucle_for,
    branche_des_if_emboites,
    instructions_executees,
    jusqu_au_break,
    premiere_branche_vraie,
    sans_les_continue,
    selon,
)
from polytech.algorithmique.domain.flux_de_sortie import Ecran, ecrire_un_reel
from polytech.algorithmique.domain.types_cpp import caractere_decale, est_vrai

TROIS = ("un", "deux", "defaut")


def test_un_if_sans_accolades_ne_commande_que_la_premiere_instruction() -> None:
    """Planches 31 et 32 : pour `i = 1`, « Il faut un i > 0 » s'affiche quand meme."""
    lignes = (" i < 0 ", " Il faut un i > 0 ", " alors!! ")
    faux, vrai = False, True
    assert instructions_executees(faux, lignes, accolades=0) == lignes[1:]
    assert instructions_executees(faux, lignes, accolades=2) == lignes[2:]
    assert instructions_executees(vrai, lignes, accolades=0) == lignes
    assert instructions_executees(vrai, lignes, accolades=2) == lignes


def test_le_else_se_rattache_au_if_le_plus_proche() -> None:
    """Planche 35 : sans accolades, `i = 5` affiche « i < 0 », et `i = -5` rien."""
    for i, sans, avec in (
        (-5, None, "sinon"),
        (5, "sinon", None),
        (2000, "alors", "alors"),
    ):
        proche = branche_des_if_emboites(i >= 0, i > 1000, accolades=False)
        ferme = branche_des_if_emboites(i >= 0, i > 1000, accolades=True)
        assert (proche, ferme) == (sans, avec)


def test_la_cascade_de_if_de_la_planche_36() -> None:
    """Majuscule, minuscule, ou ni l'un ni l'autre : la premiere condition vraie."""
    for caractere, attendue in (("G", 0), ("g", 1), ("7", 2)):
        conditions = ("A" <= caractere <= "Z", "a" <= caractere <= "z")
        assert premiere_branche_vraie(conditions) == attendue
    assert caractere_decale("G", ord("a") - ord("A")) == "g"
    assert caractere_decale("g", ord("A") - ord("a")) == "G"


def test_les_deux_boucles_for_de_la_planche_38() -> None:
    """`i` de 0 a 2, puis `j` de 5 a 1 ; la condition est evaluee avant chaque tour."""
    croissante = boucle_for(0, lambda i: i < 3, lambda i: i + 1)
    assert croissante.valeurs == (0, 1, 2)
    assert croissante.sortie == 3
    decroissante = boucle_for(5, lambda j: j > 0, lambda j: j - 1)
    assert decroissante.valeurs == (5, 4, 3, 2, 1)
    assert decroissante.sortie == 0


def test_une_condition_fausse_d_emblee_n_execute_rien() -> None:
    """Planche 40 : `for (i = 0; i > 3; i++)` n'affiche rien."""
    vide = boucle_for(0, lambda i: i > 3, lambda i: i + 1)
    assert vide.valeurs == ()
    assert vide.sortie == 0


def test_la_variable_garde_sa_valeur_de_sortie() -> None:
    """Planche 40 : apres la boucle `i < 3`, « valeur de 2xi = 6 »."""
    boucle = boucle_for(0, lambda i: i < 3, lambda i: i + 1)
    assert 2 * boucle.sortie == 6


def test_les_trois_boucles_de_la_planche_41() -> None:
    """« Aaa Bbb Ccc Ddd Eee », puis 0 a 0.4, puis les tables de 2 et de 3."""
    lettres = boucle_for("A", lambda c: c <= "E", lambda c: caractere_decale(c, 1))
    ecran = Ecran()
    for lettre in lettres.valeurs:
        minuscule = caractere_decale(lettre, 32)
        ecran.ecrire(lettre).ecrire(minuscule).ecrire(minuscule).ecrire(" ")
    assert ecran.texte == "Aaa Bbb Ccc Ddd Eee "
    pourcents = boucle_for(0.0, lambda p: p < 0.5, lambda p: p + 0.1)
    assert [ecrire_un_reel(p) for p in pourcents.valeurs] == [
        "0",
        "0.1",
        "0.2",
        "0.3",
        "0.4",
    ]
    tables = [
        f" {i} * {j} = {i * j}"
        for i in boucle_for(2, lambda i: i <= 3, lambda i: i + 1).valeurs
        for j in boucle_for(0, lambda j: j <= 3, lambda j: j + 1).valeurs
    ]
    assert tables[3] == " 2 * 3 = 6"
    assert tables[-1] == " 3 * 3 = 9"
    assert len(tables) == 8


def test_la_somme_cumulee_de_la_planche_43() -> None:
    """`while (i < 5)` : « Som = 1 », 3, 6, 10."""
    sommes, somme = [], 0
    for i in boucle_for(1, lambda i: i < 5, lambda i: i + 1).valeurs:
        somme += i
        sommes.append(somme)
    assert sommes == [1, 3, 6, 10]


def test_le_while_a_deux_conditions_de_la_planche_44() -> None:
    """On saisit E, T, F : la boucle s'arrete sur F, « Exterieur 3 »."""
    saisies = "ETFZ"
    etat = boucle_for(
        (0, "?"),
        lambda e: e[0] < 5 and e[1] != "F",
        lambda e: (e[0] + 1, saisies[e[0]]),
    )
    assert [e[0] for e in etat.valeurs] == [0, 1, 2]
    assert etat.sortie == (3, "F")


def test_le_do_while_s_execute_au_moins_une_fois() -> None:
    """Planche 46 : `j` de 5 a 1 ; et un tour meme si la condition est fausse."""
    decompte = boucle_do_while(5, lambda j: j > 0, lambda j: j - 1)
    assert decompte.valeurs == (5, 4, 3, 2, 1)
    une_fois = boucle_do_while(0, lambda j: j > 0, lambda j: j - 1)
    assert une_fois.valeurs == (0,)
    jamais = boucle_for(0, lambda j: j > 0, lambda j: j - 1)
    assert jamais.valeurs == ()


def test_les_trois_boucles_equivalentes_de_la_planche_48() -> None:
    """`for`, `while` et `do while` affichent tous `i = 0` a `i = 3`."""
    pre_testee = boucle_for(0, lambda i: i < 4, lambda i: i + 1)
    post_testee = boucle_do_while(0, lambda i: i < 4, lambda i: i + 1)
    assert pre_testee.valeurs == post_testee.valeurs == (0, 1, 2, 3)


def test_une_boucle_dont_la_variable_n_avance_pas_est_infinie() -> None:
    """Planches 38 et 44 : `for ( ; ; )`, et `while (i < 5);` au `;` de trop."""
    with pytest.raises(BoucleInfinie, match="boucle infinie"):
        boucle_for(1, lambda i: i < 5, lambda i: i)
    with pytest.raises(BoucleInfinie, match="boucle infinie"):
        boucle_do_while(0, lambda _: True, lambda i: i + 1)


def test_break_et_continue_a_la_planche_50() -> None:
    """`break` en 0 : -2, -1, 0 puis « Fin » ; `continue` en 0 : -2, -1, 1, 2."""
    valeurs = boucle_for(-2, lambda i: i <= 2, lambda i: i + 1).valeurs
    assert jusqu_au_break(valeurs, lambda i: i == 0) == (-2, -1, 0)
    assert sans_les_continue(valeurs, lambda i: i == 0) == (-2, -1, 1, 2)
    assert jusqu_au_break(valeurs, lambda i: i == 9) == valeurs


def test_le_break_ne_quitte_que_la_boucle_interieure() -> None:
    """Planche 51 : pour chaque `i`, `j` vaut 0 puis 1, et sa boucle s'arrete."""
    lignes = []
    for i in boucle_for(0, lambda i: i < 3, lambda i: i + 1).valeurs:
        lignes.append(f" valeur de i= {i}")
        interieure = boucle_for(0, lambda j: j < 5, lambda j: j + 1).valeurs
        lignes += [
            f" valeur de j= {j}" for j in jusqu_au_break(interieure, lambda j: j == 1)
        ]
    assert len(lignes) == 9
    assert lignes[:3] == [" valeur de i= 0", " valeur de j= 0", " valeur de j= 1"]
    assert lignes[6:] == [" valeur de i= 2", " valeur de j= 0", " valeur de j= 1"]


def test_le_switch_avec_ses_break() -> None:
    """Planche 52 : chaque cas execute son bloc, `default` prend le reste."""
    cas = (Cas(1, "un"), Cas(2, "deux"))
    assert selon(1, cas, defaut="defaut") == ("un",)
    assert selon(2, cas, defaut="defaut") == ("deux",)
    assert selon(7, cas, defaut="defaut") == ("defaut",)
    assert selon(7, cas, defaut=None) == ()


def test_un_break_oublie_laisse_tomber_dans_le_cas_suivant() -> None:
    """Sans `break`, l'execution se poursuit dans les blocs qui suivent."""
    cas = (Cas(1, "un", avec_break=False), Cas(2, "deux", avec_break=False))
    assert selon(1, cas, defaut="defaut") == TROIS
    assert selon(2, cas, defaut="defaut") == TROIS[1:]
    assert selon(7, cas, defaut="defaut") == TROIS[2:]
    mixte = (Cas(1, "un", avec_break=False), Cas(2, "deux"))
    assert selon(1, mixte, defaut="defaut") == ("un", "deux")


def test_le_menu_de_la_planche_53() -> None:
    """Rayon 3.5 : circonference « 21.9911 », surface « 38.4845 »."""
    pi, rayon = 3.14159, 3.5
    cas = (
        Cas(1, ecrire_un_reel(2 * pi * rayon)),
        Cas(2, ecrire_un_reel(pi * rayon**2)),
    )
    assert selon(1, cas, defaut="Mauvais choix") == ("21.9911",)
    assert selon(2, cas, defaut="Mauvais choix") == ("38.4845",)
    assert selon(3, cas, defaut="Mauvais choix") == ("Mauvais choix",)


def test_une_valeur_non_nulle_sert_de_condition() -> None:
    """Planche 30 : `if (i)` equivaut a `if (i != 0)`."""
    lignes = (" Alarme ",)
    for i in (-3, 0, 4):
        attendu = lignes if i != 0 else ()
        assert instructions_executees(est_vrai(i), lignes, accolades=0) == attendu
