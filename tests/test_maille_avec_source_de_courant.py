"""L'exercice 8 : Thevenin d'un circuit a source de courant dans la maille.

Trois difficultes que l'exercice 1 n'avait pas, et le corrige doit les
nommer :

* le **condensateur**. En regime stationnaire il ne passe aucun courant,
  donc se comporte en circuit ouvert -- et c'est ce qui simplifie le
  circuit avant tout calcul. Un corrige qui commencerait par les mailles
  sans cette remarque calculerait dans un circuit qui n'existe pas ;
* la **resistance de Thevenin avec une source de courant**. Eteindre une
  source de tension la court-circuite ; eteindre une source de courant
  l'**ouvre**. La confusion donnerait `R_1 + R_2` au lieu du parallele ;
* le **courant de maille**, qui n'est pas celui de la source. La source
  impose `I` dans sa branche, et le courant `I_1` de la maille s'en
  deduit par la loi des mailles.

Et ce fichier consigne une faute du manuscrit, que le recalcul a trouvee.
La transcription ecrit

    E_TH = 10 + 5 x (5/6 + 1) = 20 + 5 x 11/6 = 19,16 V

Le membre du milieu vaut `29,17`, non `19,16` : le `20` y est un lapsus
pour `10`. Le resultat annonce est juste, le chemin ecrit ne l'est pas --
et c'est le genre d'ecart qu'une relecture ne voit pas, parce que la
conclusion est bonne.
"""

from __future__ import annotations

import pytest

from polytech.electromagnetisme.domain.circuits.associations import (
    courant_dans_la_charge,
    courant_de_maille,
    resistance_en_parallele,
    thevenin_de_la_maille,
)
from polytech.electromagnetisme.domain.circuits.stationnaires import Thevenin

# Donnees de la transcription.
E1, E2 = 20.0, 10.0
R1, R2 = 1.0, 5.0
COURANT_INJECTE = 1.0
CHARGE = 10.0
COURANT_DE_LA_CHARGE = 3.0


def test_la_resistance_de_thevenin_est_le_parallele_des_deux() -> None:
    """`R_TH = R_1 // R_2 = 5/6` ohm, et non leur somme.

    Le parallele vient de ce qu'eteindre la source de courant l'**ouvre** :
    sa branche disparait, et les deux resistances se retrouvent cote a
    cote entre les memes noeuds.
    """
    assert resistance_en_parallele(R1, R2) == pytest.approx(5 / 6)
    assert resistance_en_parallele(R1, R2) < min(R1, R2)


def test_un_parallele_est_toujours_plus_petit_que_le_plus_petit() -> None:
    """Propriete qui garde le calcul : deux chemins conduisent mieux qu'un.

    C'est la verification la moins chere de l'exercice, et elle attrape
    l'erreur la plus courante -- additionner au lieu de mettre en
    parallele.
    """
    for a, b in ((1.0, 5.0), (100.0, 100.0), (0.5, 1e6)):
        assert resistance_en_parallele(a, b) <= min(a, b)


def test_deux_resistances_egales_en_parallele_valent_la_moitie() -> None:
    """Cas que l'on verifie a vue, et qui fixe le facteur."""
    assert resistance_en_parallele(4.0, 4.0) == pytest.approx(2.0)


def test_le_courant_de_maille_se_deduit_de_la_loi_des_mailles() -> None:
    """`I_1 = (E_1 - E_2 - R_2 I) / (R_1 + R_2) = 5/6` A.

    Le terme `-R_2 I` est la contribution de la source de courant : elle
    traverse `R_2` en plus de `I_1`, et c'est ce qui la fait entrer dans
    la maille sans y etre une force electromotrice.
    """
    assert courant_de_maille(E1, E2, R1, R2, COURANT_INJECTE) == pytest.approx(5 / 6)


def test_sans_source_de_courant_la_maille_est_le_diviseur_ordinaire() -> None:
    """A `I = 0`, il reste `(E_1 - E_2)/(R_1 + R_2)`.

    Cas limite, et il verifie que le terme supplementaire est bien au bon
    endroit : une faute de signe le laisserait apparaitre ici.
    """
    assert courant_de_maille(E1, E2, R1, R2, 0.0) == pytest.approx(
        (E1 - E2) / (R1 + R2),
    )


def test_la_tension_de_thevenin_vaut_cent_quinze_sixiemes() -> None:
    """`E_TH = E_2 + R_2 (I_1 + I) = 115/6 ~= 19,17` V.

    C'est la valeur que le manuscrit annonce, et le module y arrive par
    le chemin que le manuscrit decrit -- non par celui qu'il **ecrit**,
    dont le membre du milieu porte un lapsus.
    """
    thevenin = thevenin_de_la_maille(E1, E2, R1, R2, COURANT_INJECTE)
    assert thevenin.tension_a_vide == pytest.approx(115 / 6)
    assert thevenin.resistance == pytest.approx(5 / 6)


def test_le_lapsus_du_manuscrit_est_bien_un_lapsus() -> None:
    """Le membre du milieu du manuscrit ne vaut pas ce qu'il conclut.

    `20 + 5 x 11/6` vaut `29,17` et non `19,16`. Le `20` est un lapsus
    pour `10` -- le resultat annonce est juste, le chemin ecrit ne l'est
    pas. Ce test fixe l'ecart pour qu'il ne se reintroduise pas dans le
    corrige par recopie.
    """
    ecrit_par_le_manuscrit = 20 + 5 * 11 / 6
    juste = thevenin_de_la_maille(E1, E2, R1, R2, COURANT_INJECTE).tension_a_vide
    assert ecrit_par_le_manuscrit == pytest.approx(29.1667, abs=1e-3)
    assert juste == pytest.approx(19.1667, abs=1e-3)
    assert ecrit_par_le_manuscrit != pytest.approx(juste)


def test_le_courant_dans_la_charge_vaut_exactement_deux_amperes() -> None:
    """`I = (E_TH + R_TH I') / (R' + R_TH) = 2` A, exactement.

    L'exactitude n'est pas une coincidence de l'enonce : les sixiemes de
    `E_TH` et de `R_TH` se compensent, et c'est ce qui fait un resultat
    entier a partir de deux fractions. Un corrige qui arrondirait
    `E_TH` a `19,17` avant cette etape rendrait `2,0003`.
    """
    thevenin = thevenin_de_la_maille(E1, E2, R1, R2, COURANT_INJECTE)
    assert courant_dans_la_charge(thevenin, CHARGE, COURANT_DE_LA_CHARGE) == (
        pytest.approx(2.0)
    )


def test_arrondir_avant_la_derniere_etape_degraderait_le_resultat() -> None:
    """La raison pour laquelle le module calcule en flottants jusqu'au bout.

    Avec `E_TH` arrondi a deux decimales, le courant ne vaut plus deux
    amperes exactement. L'ecart est faible, mais il transforme un
    resultat **exact** en resultat approche -- et c'est une information
    que l'exercice donne gratuitement.
    """
    arrondi = Thevenin(tension_a_vide=19.17, resistance=5 / 6)
    approche = courant_dans_la_charge(arrondi, CHARGE, COURANT_DE_LA_CHARGE)
    assert approche != pytest.approx(2.0, abs=1e-6)
    assert approche == pytest.approx(2.0, abs=1e-3)


def test_une_resistance_nulle_en_parallele_est_refusee() -> None:
    """Un court-circuit n'est pas une resistance : le parallele s'annule."""
    with pytest.raises(ValueError, match="resistance nulle ou negative"):
        resistance_en_parallele(0.0, 5.0)


def test_une_somme_de_resistances_nulle_est_refusee_dans_la_maille() -> None:
    """La loi des mailles y diviserait par zero."""
    with pytest.raises(ValueError, match="resistance nulle ou negative"):
        courant_de_maille(E1, E2, 0.0, 0.0, 1.0)
