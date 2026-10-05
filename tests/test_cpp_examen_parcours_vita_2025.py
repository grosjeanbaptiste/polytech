"""Examen du 13 janvier 2025, « Parcours Vita 2025 » : son exemple d'execution.

Source : `ProgrammationAlgorithmique/EXAMENS/
Exam_2025_Projet_Parcours_Vita.pdf`, deux pages. La seconde imprime un
exemple d'execution complet -- choix 1, trois choix 2, choix 3, deux choix
4, choix 5 -- et c'est lui qui sert d'oracle, colonne par colonne.

L'exemple n'est pas lineaire : apres le premier choix 4, la colonne L1
garde « Nico 28 » et « Anne 28 » alors que le second choix 4 les ote. Les
deux suppressions sont donc rejouees ici dans l'ordre de la page.
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.repartition import entrelacer, partager
from polytech.algorithmique.domain.tableaux_de_structures import (
    meilleurs,
    sans_ceux,
)

PARTICIPANT_TXT = (
    "1;Jean;30",
    "2;Marie;25",
    "2;Anne;28",
    "1;Michel;20",
    "1;Isabelle;19",
    "2;Flavio;19",
    "1;Lucas;21",
    "2;Alexis;24",
    "1;Eva;20",
)
L1 = [("Jean", 30), ("Michel", 20), ("Isabelle", 19), ("Lucas", 21), ("Eva", 20)]
V2 = [("Marie", 25), ("Anne", 28), ("Flavio", 19), ("Alexis", 24)]
Participant = tuple[str, int]
Rang = list[Participant]


def _age(participant: Participant) -> int:
    return participant[1]


def _apres_les_trois_ajouts_et_la_permutation() -> tuple[Rang, Rang]:
    l1 = [*L1, ("Nicolas", 28), ("Aline", 23)]
    v2 = [*V2, ("Nico", 28)]
    return v2, l1


def _oter_les_plus_ages(
    l1: Rang, v2: Rang
) -> tuple[Rang, Rang, tuple[Participant, ...]]:
    doyens = meilleurs([*l1, *v2], _age)
    return (
        list(sans_ceux(l1, lambda p: p in doyens)),
        list(sans_ceux(v2, lambda p: p in doyens)),
        doyens,
    )


def test_le_choix_1_partage_le_fichier_entre_la_liste_et_le_vecteur() -> None:
    """« Si Val = 1, le participant sera stocke dans la List L1 » ; 2, dans V2."""
    partage = partager(PARTICIPANT_TXT)
    assert partage == {1: L1, 2: V2}


def test_le_partage_garde_l_ordre_du_fichier() -> None:
    """« Insertion a la fin » : Jean avant Michel, Marie avant Anne."""
    partage = partager(("2;B;1", "1;A;2", "2;C;3"))
    assert partage == {2: [("B", 1), ("C", 3)], 1: [("A", 2)]}


@pytest.mark.parametrize(
    "ligne", ["1;Jean", "1;Jean;30;x", "Jean;30", "1;Jean;trente", "un;Jean;30"]
)
def test_une_ligne_mal_formee_est_refusee(ligne: str) -> None:
    """« Les lignes du fichier s'agencent comme suit : Val;Prenom;Age »."""
    with pytest.raises(ValueError, match="ligne mal formee, Val;Prenom;Age attendu"):
        partager((ligne,))


def test_une_valeur_autre_que_1_ou_2_est_refusee() -> None:
    """« Val = 1 ou 2 » : il n'y a pas de troisieme conteneur."""
    with pytest.raises(ValueError, match="Val vaut 1 ou 2"):
        partager(("3;Jean;30",))


def test_le_choix_3_permute_les_deux_conteneurs() -> None:
    """Apres permutation, L1 commence par Marie et V2 finit par Aline."""
    l1, v2 = _apres_les_trois_ajouts_et_la_permutation()
    assert [prenom for prenom, _ in l1] == ["Marie", "Anne", "Flavio", "Alexis", "Nico"]
    assert v2[0] == ("Jean", 30)
    assert v2[-2:] == [("Nicolas", 28), ("Aline", 23)]


def test_le_premier_choix_4_ote_jean() -> None:
    """« Les plus ages sont : Jean 30 dans V2 »."""
    l1, v2, doyens = _oter_les_plus_ages(*_apres_les_trois_ajouts_et_la_permutation())
    assert doyens == (("Jean", 30),)
    assert len(l1) == 5
    assert [prenom for prenom, _ in v2] == [
        "Michel",
        "Isabelle",
        "Lucas",
        "Eva",
        "Nicolas",
        "Aline",
    ]


def test_le_second_choix_4_ote_trois_participants_du_meme_age() -> None:
    """« Anne 28 dans L1, Nico 28 dans L1, Nicolas 28 dans V2 »."""
    l1, v2, _ = _oter_les_plus_ages(*_apres_les_trois_ajouts_et_la_permutation())
    l1, v2, doyens = _oter_les_plus_ages(l1, v2)
    assert doyens == (("Anne", 28), ("Nico", 28), ("Nicolas", 28))
    assert l1 == [("Marie", 25), ("Flavio", 19), ("Alexis", 24)]
    assert v2 == [
        ("Michel", 20),
        ("Isabelle", 19),
        ("Lucas", 21),
        ("Eva", 20),
        ("Aline", 23),
    ]


def test_le_choix_5_entrelace_les_prenoms_en_commencant_par_la_liste() -> None:
    """« Marie / Michel / Flavio / Isabelle / Alexis / Lucas / Eva / Aline »."""
    l1 = ["Marie", "Flavio", "Alexis"]
    v2 = ["Michel", "Isabelle", "Lucas", "Eva", "Aline"]
    assert " / ".join(entrelacer(l1, v2)) == (
        "Marie / Michel / Flavio / Isabelle / Alexis / Lucas / Eva / Aline"
    )


def test_le_choix_3_puis_le_choix_5() -> None:
    """« Michel / Marie / Isabelle / Flavio / Lucas / Alexis / Eva / Aline »."""
    l1 = ["Michel", "Isabelle", "Lucas", "Eva", "Aline"]
    v2 = ["Marie", "Flavio", "Alexis"]
    assert " / ".join(entrelacer(l1, v2)) == (
        "Michel / Marie / Isabelle / Flavio / Lucas / Alexis / Eva / Aline"
    )


def test_entrelacer_avec_un_conteneur_vide_rend_l_autre() -> None:
    """Rien a alterner : l'ordre de l'autre conteneur est garde."""
    assert entrelacer([], ["a", "b"]) == ["a", "b"]
    assert entrelacer(["a", "b"], []) == ["a", "b"]
    assert entrelacer([], []) == []
