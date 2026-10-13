"""Le recueil de primitives du syllabus de 2011 : onze exercices, un compte tenu.

`Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf` porte, pages 60 a
63 du PDF, un chapitre « Primitives » de onze exercices -- cent vingt-quatre
primitives -- dont il n'imprime que les reponses (pages 64 a 72). Le
document est un scan : ses exercices n'ont pas de titre dans le texte
converti, et aucune mesure du depot ne les voyait.

Ce test les nomme un a un, avec le dossier qui les corrige. Le cliquet
ne fait que descendre.
"""

from __future__ import annotations

from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
TRAVAUX = RACINE / "travaux" / "Mathematique"

EXERCICES: dict[str, tuple[int, str | None]] = {
    "1. Methodes classiques": (56, "recueil-primitives-1"),
    "2. Fractions rationnelles": (8, "recueil-primitives-2"),
    "3. Recurrence sur 1 / (x^2 + 1)^n": (3, "recueil-primitives-2"),
    "4. Exponentielles": (4, "recueil-primitives-2"),
    "5. Fonctions diverses": (5, "recueil-primitives-3"),
    "6. Fonctions diverses, suite": (8, "recueil-primitives-3"),
    "7. Fonctions diverses, fin": (5, "recueil-primitives-3"),
    "8. Rationnelles en sin x et cos x": (16, "recueil-primitives-4"),
    "9. Irrationnelles simples": (4, "recueil-primitives-5"),
    "10. Racine d'un trinome du second degre": (6, "recueil-primitives-5"),
    "11. Revision": (9, "recueil-primitives-5"),
}
"""Exercice, dans l'ordre des pages : le nombre de primitives, et le dossier."""

PRIMITIVES_DU_RECUEIL = 124
EXERCICES_SANS_CORRIGE = 0
PRIMITIVES_SANS_CORRIGE = 0


def test_le_recueil_compte_cent_vingt_quatre_primitives() -> None:
    assert sum(nombre for nombre, _ in EXERCICES.values()) == PRIMITIVES_DU_RECUEIL


def test_chaque_dossier_nomme_existe_et_declare_le_syllabus() -> None:
    for _, dossier in EXERCICES.values():
        if dossier is None:
            continue
        declaration = (TRAVAUX / dossier / "activite.toml").read_text("utf-8")
        assert "Syllabus_Notes_de_cours_2011.pdf" in declaration


def test_le_cliquet_des_exercices_restants() -> None:
    restants = [titre for titre, (_, dossier) in EXERCICES.items() if dossier is None]
    assert len(restants) == EXERCICES_SANS_CORRIGE


def test_le_cliquet_des_primitives_restantes() -> None:
    restantes = sum(n for n, dossier in EXERCICES.values() if dossier is None)
    assert restantes == PRIMITIVES_SANS_CORRIGE
