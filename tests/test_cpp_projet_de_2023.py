"""Projet IG Charleroi de 2023 : `map` et `multimap`.

Source : `ProgrammationAlgorithmique/EXAMENS/Exam_2023_Projet.pdf`, sept
pages. La page 3 donne un programme et sa « solution » (figure 2), les
pages 5 et 6 le menu du projet : chercher par cle, chercher tous les pays
d'une region, supprimer toutes les lignes d'une region, effacer.

Un ecart, constate sur la page rendue : le programme imprime n'insere que
deux paires, `souris` et `elephant`, alors que la figure 2 affiche « Map
size : 4 » avec `chat --> 3` et `tigre --> 200`. Deux insertions manquent
au listing. Les quatre paires de la figure sont celles que ce test rejoue.
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.associatifs import Associatif


def _poids_de_la_figure_2() -> Associatif:
    poids = Associatif("map")
    poids.affecter("souris", 1)
    poids.insert("elephant", 10000)
    poids.insert("tigre", 200)
    poids.insert("chat", 3)
    return poids


def _pays() -> Associatif:
    pays = Associatif("multimap")
    for region, nom in (
        ("WESTERN EUROPE", "Belgium"),
        ("BALTICS", "Estonia"),
        ("WESTERN EUROPE", "France"),
        ("BALTICS", "Latvia"),
        ("OCEANIA", "Fiji"),
    ):
        pays.insert(region, nom)
    return pays


def test_la_premiere_moitie_de_la_figure_2() -> None:
    """« Map size : 4 », puis chat, elephant, souris, tigre : l'ordre des cles."""
    poids = _poids_de_la_figure_2()
    assert poids.size() == 4
    assert [(cle, poids.at(cle)) for cle in poids.cles()] == [
        ("chat", 3),
        ("elephant", 10000),
        ("souris", 1),
        ("tigre", 200),
    ]


def test_l_incrementation_de_la_figure_2() -> None:
    """`poids["souris"]++` : « souris --> 2 », et toujours quatre elements."""
    poids = _poids_de_la_figure_2()
    poids.affecter("souris", poids.at("souris") + 1)
    assert poids.at("souris") == 2
    assert poids.size() == 4


def test_les_crochets_ajoutent_ou_ecrasent() -> None:
    """Page 3 : « ajoute ou ecrase la valeur existante » ; `insert`, lui, refuse."""
    poids = Associatif("map")
    poids.affecter("souris", 1)
    assert not poids.insert("souris", 5)
    assert poids.at("souris") == 1
    poids.affecter("souris", 5)
    assert (poids.at("souris"), poids.size()) == (5, 1)


def test_clear_vide_le_conteneur() -> None:
    """Figure 2, derniere ligne : « Map size : 0 »."""
    poids = _poids_de_la_figure_2()
    poids.clear()
    assert poids.size() == 0
    assert poids.cles() == ()


def test_le_multimap_est_range_par_region() -> None:
    """Nota bene de la page 6 : « ordonnees par ordre alphabetique de la cle »."""
    assert _pays().cles() == (
        "BALTICS",
        "BALTICS",
        "OCEANIA",
        "WESTERN EUROPE",
        "WESTERN EUROPE",
    )


def test_tous_les_pays_d_une_region() -> None:
    """Choix 4 : « chercher tous les pays situes dans une region »."""
    pays = _pays()
    assert pays.valeurs_de("BALTICS") == ("Estonia", "Latvia")
    assert pays.valeurs_de("WESTERN EUROPE") == ("Belgium", "France")
    assert pays.valeurs_de("NEAR EAST") == ()


def test_supprimer_toutes_les_lignes_d_une_region() -> None:
    """Choix 8 : `erase(cle)` ote toutes les lignes et rend leur nombre."""
    pays = _pays()
    assert pays.erase("BALTICS") == 2
    assert pays.cles() == ("OCEANIA", "WESTERN EUROPE", "WESTERN EUROPE")
    assert pays.valeurs_de("WESTERN EUROPE") == ("Belgium", "France")
    assert pays.erase("BALTICS") == 0
    assert pays.size() == 3


def test_les_crochets_n_existent_que_pour_le_map() -> None:
    """Une cle de multimap ne designe pas une valeur unique."""
    with pytest.raises(AttributeError, match="n'est pas membre de std::"):
        _pays().affecter("OCEANIA", "Samoa")
