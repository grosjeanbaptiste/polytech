"""Syllabus « Introduction a la programmation en C++ » de 2010-2011.

Source : `ProgrammationAlgorithmique/SYLLABUS/
Syllabus_Introduction_au_C_2010_2011.pdf`, 149 pages. C'est l'edition
anterieure du syllabus de 2015 : compare ligne a ligne, son texte est le
meme, moins le chapitre des listes chainees, plus un passage sur les
descripteurs `%d` (page 16) et, en annexe B, les quatre-vingts planches du
cours de 2011. Seul ce qui lui est propre figure ici.

Chaque affichage a ete confirme en compilant le programme (clang++ 21). La
compilation etablit un ecart :

* page 16 : `cout << ("%d", n)` est donne pour un affichage « sous un
  format ». Les parentheses renferment l'operateur virgule : l'expression
  vaut `n`, et le descripteur `"%d"` est evalue puis jete. La colonne de
  droite affiche donc exactement ce qu'affiche celle de gauche.
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.controle import boucle_do_while, boucle_for
from polytech.algorithmique.domain.expressions import evaluer
from polytech.algorithmique.domain.flux_de_sortie import ecrire_un_reel
from polytech.algorithmique.domain.memoire import Memoire
from polytech.algorithmique.domain.types_cpp import (
    caractere_decale,
    en_simple_precision,
    modulo,
)


def test_l_operateur_virgule_rend_son_operande_de_droite() -> None:
    """Ecart de la page 16 : `("%d", n)` vaut `n` ; annexe A, le moins prioritaire."""
    assert evaluer("5, 7") == 7
    assert evaluer("(1, 2) + 3") == 5
    assert evaluer("1 + 1, 2 * 3, 4") == 4


def test_la_virgule_evalue_ses_operandes_de_gauche_a_droite() -> None:
    """Le premier operande est evalue, puis jete : une division par zero s'y voit."""
    with pytest.raises(ZeroDivisionError, match="division par zero"):
        evaluer("1 / 0, 7")
    with pytest.raises(SyntaxError, match="expression mal formee"):
        evaluer("5 ,")


def test_un_float_n_a_que_la_simple_precision() -> None:
    """Page 8 : `float`, reel simple precision sur 4 octets ; `double`, sur 8."""
    assert en_simple_precision(0.1) != 0.1
    assert en_simple_precision(0.5) == 0.5
    assert ecrire_un_reel(en_simple_precision(0.1), 9) == "0.100000001"


def test_la_boucle_des_pourcentages_de_la_planche_25() -> None:
    """`float pourcent ; pourcent < 0.8 ; pourcent += 0.1` : huit tours, de 0 a 0.7."""
    tours = boucle_for(
        0.0, lambda p: p < 0.8, lambda p: en_simple_precision(p + 0.1)
    ).valeurs
    assert [ecrire_un_reel(p) for p in tours] == [
        "0",
        "0.1",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
        "0.6",
        "0.7",
    ]


def test_la_meme_boucle_en_double_fait_un_tour_de_plus() -> None:
    """En double precision, huit additions de 0.1 restent sous 0.8 : neuf tours."""
    tours = boucle_for(0.0, lambda p: p < 0.8, lambda p: p + 0.1).valeurs
    assert (len(tours), ecrire_un_reel(tours[-1])) == (9, "0.8")


def test_les_deux_boucles_sur_les_lettres_de_la_planche_25() -> None:
    """De A a M en montant, de z a n en descendant."""
    montee = boucle_for("A", lambda c: c <= "M", lambda c: caractere_decale(c, 1))
    descente = boucle_for("z", lambda c: c >= "n", lambda c: caractere_decale(c, -1))
    assert "".join(montee.valeurs) == "ABCDEFGHIJKLM"
    assert "".join(descente.valeurs) == "zyxwvutsrqpon"


def test_le_while_dont_la_condition_est_fausse_d_emblee() -> None:
    """Planche 23 : `int j = 5; while (j == 0)` -- « start », puis « end »."""
    assert boucle_for(5, lambda j: j == 0, lambda j: j - 1).valeurs == ()


def test_le_do_while_de_la_planche_24() -> None:
    """« j = 5 » a « j = 1 », puis « stop »."""
    parcours = boucle_do_while(5, lambda j: j > 0, lambda j: j - 1)
    assert (parcours.valeurs, parcours.sortie) == ((5, 4, 3, 2, 1), 0)


def test_la_parite_par_le_modulo_de_la_planche_15() -> None:
    """`4%2 = 0`, `5%2 = 1`, `8%3 = 2` : `A%2` nul, A est pair."""
    assert (modulo(4, 2), modulo(5, 2), modulo(8, 3)) == (0, 1, 2)
    assert (evaluer("5 / 4"), evaluer("5.0 / 4.0")) == (1, 1.25)


def test_les_trois_sommes_de_carres_de_la_planche_42() -> None:
    """`Fnct_SomCar(5)`, `(10)` et `(7)` : 30, 285 et 91."""
    assert [sum(i * i for i in range(n)) for n in (5, 10, 7)] == [30, 285, 91]


def test_les_deux_parametres_par_reference_de_la_planche_62() -> None:
    """`v1 *= 10; v2 += 7;` par reference : « var1 = 50 var2 = 7 »."""
    memoire = Memoire()
    memoire.declarer("var1", 5)
    memoire.declarer("var2", 0)
    memoire.referencer("v1", "var1")
    memoire.referencer("v2", "var2")
    memoire.affecter("v1", memoire.valeur("v1") * 10)
    memoire.affecter("v2", memoire.valeur("v2") + 7)
    assert (memoire.valeur("var1"), memoire.valeur("var2")) == (50, 7)
