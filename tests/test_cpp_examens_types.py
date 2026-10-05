"""Programmation, seance 3 : les trois examens types sur un tableau de structures.

Source : `ProgrammationAlgorithmique/SLIDES/Slides_2025_2026_S3.pdf`,
planches 42 a 44 (numerotees 104 a 106 dans le pied de page). Les trois
enonces sont des images : ils ont ete lus sur la page rendue, et chaque
tableau d'exemple est recopie ici tel quel.

* planche 42, « Villes belges les plus peuplees » (2 h, `Nmax = 22`) ;
* planche 43, le tableau de `Personne` (`Nmax = 6`) ;
* planche 44, le tableau d'`Etudiant` et de ses cotes (`NE_max = 6`,
  `NC_max = 5`).

Les positions des enonces se comptent a partir de 1 ; le tableau du
domaine, comme celui du C++, a partir de 0.
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.tableaux import TableauStatique
from polytech.algorithmique.domain.tableaux_de_structures import (
    exiger_strictement_entre,
    extremes,
    meilleurs,
    rang_qui_suit,
    sans_ceux,
    un_sur_deux,
)

VILLES = [("Anvers", 460, "f"), ("La Louviere", 77, "w"), ("Bruxelles-ville", 143, "b")]
PERSONNES = [("Riri", 20), ("Fifi", 40), ("Lou Lou", 25)]
ETUDIANTS = [("Van Ix", (20, 9)), ("Obelix", (20, 5)), ("Amerix", (15, 0))]


def _nom(element: tuple[str, object]) -> str:
    return element[0]


def test_l_ajout_des_villes_a_la_position_donnee() -> None:
    """Planche 42 : Mons en tete, puis Bruges en position 5 -- la derniere."""
    villes = TableauStatique(22, list(VILLES))
    villes.inserer(0, ("Mons", 91, "w"))
    villes.inserer(5 - 1, ("Bruges", 117, "f"))
    assert [_nom(v) for v in villes.elements] == [
        "Mons",
        "Anvers",
        "La Louviere",
        "Bruxelles-ville",
        "Bruges",
    ]


def test_la_ville_la_plus_peuplee_et_la_moins_peuplee() -> None:
    """Planche 42 : Anvers, 460 ; La Louviere, 77."""
    plus, moins = extremes(VILLES, lambda ville: ville[1])
    assert (plus, moins) == (("Anvers", 460, "f"), ("La Louviere", 77, "w"))


def test_le_nombre_de_e_des_deux_villes() -> None:
    """Planche 42 : « Anvers 460 1 (e) » et « La Louviere 77 2 (e) »."""
    plus, moins = extremes(VILLES, lambda ville: ville[1])
    assert (_nom(plus).count("e"), _nom(moins).count("e")) == (1, 2)


def test_entre_deux_ex_aequo_les_extremes_gardent_le_premier() -> None:
    """La recherche du maximum par `>` strict ne remplace pas un egal."""
    paires = [("a", 3), ("b", 7), ("c", 7), ("d", 3)]
    assert extremes(paires, lambda p: p[1]) == (("b", 7), ("a", 3))


def test_un_tableau_vide_n_a_pas_d_extremes() -> None:
    """Ni plus peuplee ni moins peuplee."""
    with pytest.raises(ValueError, match="tableau vide"):
        extremes([], len)
    with pytest.raises(ValueError, match="tableau vide"):
        meilleurs([], len)


def test_a_la_position_p_tu_ajouteras() -> None:
    """Planche 43 : Tata en tete, puis MeToo en position 3."""
    personnes = TableauStatique(6, list(PERSONNES))
    personnes.inserer(0, ("Tata", 55))
    personnes.inserer(3 - 1, ("MeToo", 18))
    assert personnes.elements == [
        ("Tata", 55),
        ("Riri", 20),
        ("MeToo", 18),
        ("Fifi", 40),
        ("Lou Lou", 25),
    ]


def test_l_ajout_s_arrete_quand_le_tableau_est_plein() -> None:
    """Planche 43, le panneau « N < Nmax » : six personnes, pas une de plus."""
    personnes = TableauStatique(6, list(PERSONNES))
    for nom in ("Tata", "MeToo", "Zaza"):
        personnes.ajouter_a_la_fin((nom, 30))
    with pytest.raises(OverflowError, match="tableau plein"):
        personnes.ajouter_a_la_fin(("Toto", 30))


def test_un_sur_deux_tu_afficheras() -> None:
    """Planche 43 : Tata 55 4, MeToo 18 5, Lou Lou 25 7."""
    tableau = [("Tata", 55), ("Riri", 20), ("MeToo", 18), ("Fifi", 40), ("Lou Lou", 25)]
    affiches = un_sur_deux(tableau)
    assert affiches == (("Tata", 55), ("MeToo", 18), ("Lou Lou", 25))
    assert [len(_nom(p)) for p in affiches] == [4, 5, 7]
    assert un_sur_deux(tableau[:4]) == (("Tata", 55), ("MeToo", 18))


def test_la_valeur_v_doit_etre_entre_dix_et_cent() -> None:
    """Planche 43 : « Si V = 120, on doit signaler que 120 est > 100 »."""
    with pytest.raises(ValueError, match="trop grande"):
        exiger_strictement_entre(120, 10, 100)
    with pytest.raises(ValueError, match="trop petite"):
        exiger_strictement_entre(10, 10, 100)
    with pytest.raises(ValueError, match="trop grande"):
        exiger_strictement_entre(100, 10, 100)
    exiger_strictement_entre(11, 10, 100)
    exiger_strictement_entre(99, 10, 100)


def test_superieur_a_v_tu_supprimeras() -> None:
    """Planche 43 : toute personne dont l'age est strictement superieur a V."""
    tableau = [("Tata", 55), ("Riri", 20), ("MeToo", 18), ("Fifi", 40), ("Lou Lou", 25)]
    assert sans_ceux(tableau, lambda p: p[1] > 25) == (
        ("Riri", 20),
        ("MeToo", 18),
        ("Lou Lou", 25),
    )
    assert sans_ceux(tableau, lambda p: p[1] > 55) == tuple(tableau)


def test_ajout_apres_un_etudiant_qui_n_existe_pas() -> None:
    """Planche 44 : « Desole Titeuf n'existe pas, donc pas d'ajout »."""
    with pytest.raises(LookupError, match="existe pas, donc pas d'ajout"):
        rang_qui_suit(ETUDIANTS, lambda e: _nom(e) == "Titeuf")


def test_ajout_apres_van_ix() -> None:
    """Planche 44 : Asterix, 19 et 12, vient juste apres Van Ix."""
    etudiants = TableauStatique(6, list(ETUDIANTS))
    rang = rang_qui_suit(etudiants.elements, lambda e: _nom(e) == "Van Ix")
    etudiants.inserer(rang, ("Asterix", (19, 12)))
    rang = rang_qui_suit(etudiants.elements, lambda e: _nom(e) == "Obelix")
    etudiants.inserer(rang, ("ToToix", (10, 10)))
    assert [_nom(e) for e in etudiants.elements] == [
        "Van Ix",
        "Asterix",
        "Obelix",
        "ToToix",
        "Amerix",
    ]


def test_l_ajout_apres_le_dernier_se_fait_a_la_fin() -> None:
    """Le rang rendu est alors la taille du tableau."""
    assert rang_qui_suit(ETUDIANTS, lambda e: _nom(e) == "Amerix") == len(ETUDIANTS)


def test_le_meilleur_pour_chaque_cote() -> None:
    """Planche 44 : cote[0], « Van Ix ou Obelix » ; cote[1], Asterix."""
    tableau = [
        ("Van Ix", (20, 9)),
        ("Asterix", (19, 12)),
        ("Obelix", (20, 5)),
        ("ToToix", (10, 10)),
        ("Amerix", (15, 0)),
    ]
    premiers = meilleurs(tableau, lambda e: e[1][0])
    seconds = meilleurs(tableau, lambda e: e[1][1])
    assert [_nom(e) for e in premiers] == ["Van Ix", "Obelix"]
    assert [_nom(e) for e in seconds] == ["Asterix"]


def test_supprimer_qui_a_une_cote_sous_le_seuil() -> None:
    """Planche 44, seuil 9 : au moins une cote strictement inferieure."""
    tableau = [
        ("Van Ix", (20, 9)),
        ("Asterix", (19, 12)),
        ("Obelix", (20, 5)),
        ("ToToix", (10, 10)),
        ("Amerix", (15, 0)),
    ]
    gardes = sans_ceux(tableau, lambda e: min(e[1]) < 9)
    assert [_nom(e) for e in gardes] == ["Van Ix", "Asterix", "ToToix"]
