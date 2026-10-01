"""Les deux enregistrements que les TP lisent dans des fichiers texte.

L'enonce fournit `etudiants.txt` et `profs.txt`, et leur format est
decrit : des champs separes par des points-virgules, une premiere ligne
d'entete annoncant le nombre d'enregistrements et le nombre de notes.

Ces objets ne traduisent pas le C++ de l'etudiant -- ce serait sans
interet. Ils servent d'**oracle** : ils calculent independamment ce que le
corrige devra produire, et c'est ce qui donne au corrige quelque chose a
prouver. Sans eux, un corrige ne pourrait qu'affirmer sa propre sortie.

L'entete est une redondance utile : il permet de detecter un fichier
tronque, ce qu'un corrige doit faire et ce que la lecture naive oublie.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pathlib import Path

SEPARATEUR: Final = ";"
CHAMPS_D_ETUDIANT: Final = 4
"""Nom, annee, matricule, section -- les notes viennent apres, en nombre libre."""

CHAMPS_DE_PROFESSEUR: Final = 2
"""Nom et mot de passe -- les sections viennent apres."""


@dataclass(frozen=True, slots=True)
class Etudiant:
    """Un etudiant : son identite, sa section, et ses notes."""

    nom: str
    annee: str
    matricule: int
    section: str
    notes: tuple[float, ...]

    @classmethod
    def depuis_la_ligne(cls, ligne: str) -> Etudiant:
        """Lit `nom;annee;matricule;section;note;note...`."""
        champs = ligne.strip().split(SEPARATEUR)
        if len(champs) <= CHAMPS_D_ETUDIANT:
            msg = f"ligne d'etudiant incomplete ({len(champs)} champs) : {ligne!r}"
            raise ValueError(msg)
        nom, annee, matricule, section, *notes = champs
        if not matricule.isdigit():
            msg = f"matricule non numerique : {matricule!r}"
            raise ValueError(msg)
        return cls(
            nom=nom,
            annee=annee,
            matricule=int(matricule),
            section=section,
            notes=tuple(float(note) for note in notes),
        )

    def moyenne(self) -> float:
        """La moyenne arithmetique des notes ; leve s'il n'y en a aucune."""
        if not self.notes:
            msg = f"aucune note pour {self.nom} : la moyenne n'existe pas"
            raise ValueError(msg)
        return sum(self.notes) / len(self.notes)


@dataclass(frozen=True, slots=True)
class Professeur:
    """Un professeur : son identite, et les sections qu'il enseigne."""

    nom: str
    sections: tuple[str, ...]

    @classmethod
    def depuis_la_ligne(cls, ligne: str) -> Professeur:
        """Lit `nom;mot de passe;section;section...`."""
        champs = ligne.strip().split(SEPARATEUR)
        if len(champs) <= CHAMPS_DE_PROFESSEUR:
            msg = f"aucune section pour cette ligne de professeur : {ligne!r}"
            raise ValueError(msg)
        nom, _, *sections = champs
        return cls(nom=nom, sections=tuple(sections))


def _lignes_utiles(fichier: Path) -> list[str]:
    """Les lignes non vides, entete exclue."""
    lignes = fichier.read_text(encoding="utf-8", errors="replace").splitlines()
    return [ligne for ligne in lignes[1:] if ligne.strip()]


def lire_les_etudiants(fichier: Path) -> tuple[Etudiant, ...]:
    """Les etudiants du fichier ; propage l'absence du fichier."""
    return tuple(Etudiant.depuis_la_ligne(ligne) for ligne in _lignes_utiles(fichier))


def lire_les_professeurs(fichier: Path) -> tuple[Professeur, ...]:
    """Les professeurs du fichier ; propage l'absence du fichier."""
    return tuple(Professeur.depuis_la_ligne(ligne) for ligne in _lignes_utiles(fichier))
