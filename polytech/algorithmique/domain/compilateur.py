"""Port de sortie : un compilateur C++ qui juge une source et la lance.

Le domaine dit le contrat, jamais le compilateur : l'adaptateur de
`infrastructure/` le tient (clang++, g++, ...). Un refus du compilateur
n'est pas une erreur du programme appelant : c'est un `Diagnostic`.
Ce qui l'est, c'est l'impossibilite de juger -- compilateur absent,
edition de liens impossible --, qui leve et ne se tait jamais.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class EditionDeLiensImpossible(RuntimeError):
    """La source est acceptee mais le poste ne sait pas la lier."""


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """Le verdict de l'analyse d'une source : acceptee ou non, et pourquoi."""

    accepte: bool
    messages: str

    def mentionne(self, fragment: str) -> bool:
        """Vrai si les messages du compilateur contiennent ce fragment."""
        return fragment in self.messages


class Compilateur(Protocol):
    """Contrat : diagnostiquer une source, ou la compiler et la lancer."""

    def diagnostiquer(self, source: str) -> Diagnostic:
        """Analyse seule (syntaxe et types), sans edition de liens."""
        ...

    def executer(self, source: str) -> str:
        """Compile, lie, lance ; rend la sortie standard.

        Leve `RuntimeError` si la compilation est refusee ou si le programme
        rend un code non nul, `EditionDeLiensImpossible` si le lien echoue.
        """
        ...
