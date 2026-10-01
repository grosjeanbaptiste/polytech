"""Le chapitre I de l'electromagnetisme : son histoire, et ce qu'elle ordonne.

Ce fascicule n'etait pas declare. Il annonce lui-meme ses trois parties :
historique, applications, structure du cours. Aucune des trois n'est un
calcul, et la tentation etait de l'ecarter comme « presentation » -- ce
que j'ai fait a tort pour `TH-01`, qui enseignait la loi de Newton.

Ici l'examen du fascicule donne autre chose : la chronologie **est** le
contenu, et elle est verifiable. Le precedent est `EI-0`, ou le plan du
cours s'est revele testable -- qu'un sujet soit range dans le bon
chapitre se verifie.

Et cette chronologie dit quelque chose que les chapitres suivants ne
disent pas : dans quel ordre les lois sont venues, et donc laquelle
generalise laquelle. Maxwell unifie en 1873 ce que Coulomb, Oersted,
Ampere et Faraday avaient etabli separement ; Hertz confirme en 1887 par
l'experience ce que Maxwell avait predit par le calcul. Cet ordre est
l'argument du cours, et une date fausse le romprait.
"""

from __future__ import annotations

import pytest

from polytech.electromagnetisme.domain.chronologie import (
    JALONS,
    apres,
    jalon,
    premiere_unification,
)


def test_les_jalons_sont_ordonnes_dans_le_temps() -> None:
    """L'ordre du fascicule est chronologique, et il doit le rester.

    Une date mal transcrite se verrait ici avant de se voir ailleurs :
    c'est la garde la moins couteuse de ce module.
    """
    annees = [j.annee for j in JALONS]
    assert annees == sorted(annees)


def test_la_chronologie_part_de_l_antiquite() -> None:
    """Thales de Milet, vers -625 : l'ambre frottee et la pierre de Magnesie.

    Le fascicule en fait le point de depart parce que les deux mots du
    nom de la matiere en viennent -- « elektron » pour l'ambre,
    « magnesie » pour la region d'Asie mineure.
    """
    assert JALONS[0].annee == -625
    assert "Thales" in JALONS[0].auteur


def test_maxwell_unifie_apres_les_lois_qu_il_unifie() -> None:
    """1873 vient apres Coulomb, Oersted, Ampere et Faraday, necessairement.

    C'est l'argument du cours, et non une coincidence de dates : on
    n'unifie pas ce qui n'a pas encore ete etabli. Le test l'exige de
    chacun des quatre.
    """
    unification = premiere_unification()
    assert unification.annee == 1873
    for auteur in ("Coulomb", "Oersted", "Ampere", "Faraday"):
        assert apres(unification, jalon(auteur))


def test_hertz_confirme_apres_que_maxwell_a_predit() -> None:
    """1887 apres 1873 : l'experience vient apres le calcul.

    Le fascicule y insiste -- Hertz « prouve par l'experience l'existence
    des ondes electromagnetiques » que les equations de Maxwell avaient
    fait apparaitre. L'ordre inverse serait une autre histoire des
    sciences.
    """
    assert apres(jalon("Hertz"), jalon("Maxwell"))


def test_volta_precede_les_lois_du_courant() -> None:
    """La pile de 1796 avant Oersted, Ampere, Ohm et Faraday.

    Le fascicule explique pourquoi : « il devient possible de mettre des
    charges en mouvement et de produire un courant electrique ». Sans
    source de courant continu, aucune des quatre lois suivantes n'etait
    observable -- c'est l'instrument qui a ouvert le champ.
    """
    pile = jalon("Volta")
    for auteur in ("Oersted", "Ampere", "Ohm", "Faraday"):
        assert apres(jalon(auteur), pile)


def test_chaque_jalon_nomme_ce_qu_il_a_etabli() -> None:
    """Une date sans objet ne serait qu'un nombre."""
    for j in JALONS:
        assert len(j.apport) > 10, j.auteur


def test_un_auteur_inconnu_est_refuse() -> None:
    """Chercher un jalon absent doit se dire, non rendre un jalon par defaut."""
    with pytest.raises(KeyError, match="aucun jalon"):
        jalon("Einstein")
